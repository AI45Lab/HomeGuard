"""
Model Inference Module for Risk Grounding Evaluation.

This module contains the SafetyAgent class and related inference utilities
for running model inference on safety hazard detection tasks.
"""

import base64
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from data.pipeline.utils import proxy_off, proxy_on, bbox_norm_to_pixel
from evaluation.prompt import (
    SAFETY_PRINCIPLES,
    ACTION_TRIGGER_EVAL_TEMPLATE_V1,
    ACTION_TRIGGER_EVAL_TEMPLATE_V1_WITH_COT,
    ACTION_TRIGGER_EVAL_TEMPLATE_V2,
    ACTION_TRIGGER_EVAL_TEMPLATE_V2_WITH_COT,
    ACTION_TRIGGER_EVAL_TEMPLATE_V3
)
from evaluation.utils import add_sys_path

third_party_dir = os.path.join(os.path.dirname(__file__), '..', 'third_party')
with add_sys_path(os.path.join(third_party_dir, 'Robobrain2.5')):
    from inference import UnifiedInference as RoboBrainInference

class SafetyAgent:
    """
    Model inference agent for safety hazard detection.

    Supports both local models (Qwen-VL) and API models (Gemini, GPT-4V, etc.).
    Can load LoRA adapters for local models.
    """

    def __init__(self, version, model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
                 adapter_path: Optional[str] = None,
                 device: str = "cuda", max_retries: int = 3, batch_size: int = 4):
        """
        Initialize the SafetyAgent.

        Args:
            model_name: Path to local model or name of API model
            adapter_path: Path to LoRA adapter (for local models only)
            device: Device for local model inference
            max_retries: Maximum retries for API calls
            batch_size: Batch size for local model inference
        """
        self.device = device
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.version = version

        if os.path.exists(model_name):
            self.model_type = "local"
            print(f"Loading base model: {model_name}...")
            if 'qwen' in model_name.lower():
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, device_map="auto"
                )
                self.processor = AutoProcessor.from_pretrained(model_name)
                # Set padding_side to left for decoder-only architecture
                self.processor.tokenizer.padding_side = 'left'
            elif 'robobrain' in model_name.lower():
                self.model = RoboBrainInference(model_name)
                self.processor = None
            else:
                raise ValueError(f"Unsupported local model: {model_name}")

            # Load LoRA adapter if provided
            if adapter_path:
                if os.path.exists(adapter_path):
                    print(f"Loading LoRA adapter from: {adapter_path}...")
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        adapter_path,
                        is_trainable=False
                    )
                    # Merge and unload for faster inference
                    print("Merging LoRA adapter...")
                    self.model = self.model.merge_and_unload()
                    print("LoRA adapter merged successfully.")
                else:
                    raise ValueError(f"LoRA adapter path not found: {adapter_path}")

            print("Model loaded successfully.")
        else:
            self.model_type = "api"
            self.model = model_name
            if adapter_path:
                print(f"Warning: adapter_path is ignored for API models")
            key = os.getenv("TARGET_API_KEY")
            url = os.getenv("TARGET_API_URL")
            if os.getenv("USE_API_PROXY", "0").lower() in {"1", "true", "yes", "on"}:
                proxy_on()
            else:
                proxy_off()
            self.client = OpenAI(api_key=key, base_url=url)
            print(f"Using API model: {model_name}")

    def infer_single(self, image_path: str, action: str, version: str) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Inference for a single sample.

        Args:
            image_path: Path to the image file
            action: User action
            version: Prompt version to use

        Returns:
            Tuple of (parsed_prediction, raw_output_text)
        """
        if self.model_type == "api" and "gemini" in self.model.lower():
            proxy_on()

        is_gemini_gpt = self.model_type == "api" and ("gemini" in self.model.lower() or "gpt" in self.model.lower())

        # Always use action_triggered templates
        if version.lower() == "v1":
            template = ACTION_TRIGGER_EVAL_TEMPLATE_V1
        elif version.lower() == "v1_cot":
            template = ACTION_TRIGGER_EVAL_TEMPLATE_V1_WITH_COT
        elif version.lower() == "v2":
            template = ACTION_TRIGGER_EVAL_TEMPLATE_V2
        elif version.lower() == "v2_cot":
            template = ACTION_TRIGGER_EVAL_TEMPLATE_V2_WITH_COT
        elif version.lower() == "v3":
            template = ACTION_TRIGGER_EVAL_TEMPLATE_V3
        else:
            raise NotImplementedError("Version Not Found")
        
        prompt_text = template.format(action=action, safety_principles=SAFETY_PRINCIPLES)
        if is_gemini_gpt:
            prompt_text = prompt_text.replace('x_min, y_min, x_max, y_max', 'y_min, x_min, y_max, x_max')

        if "think" in self.model_name.lower():
            max_new_tokens = 8192
        else:
            max_new_tokens = 512

        if self.model_type == "local":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
        
            if self.processor is not None:
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.model.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            else: 
                output_text = self.model.inference(text=prompt_text, image=image_path, task='general')['answer']
        else:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            output_text = None
            for attempt in range(self.max_retries):
                try:
                    res = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0
                    )
                    output_text = res.choices[0].message.content
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
                    else:
                        return None, f"API Error: {e}"

        return output_text

    def infer_batch(self, items: List[Dict]) -> List[Dict]:
        """
        Batch inference for local model or parallel API calls.

        Args:
            items: List of dicts with keys: id, image_path, action, hazard_type

        Returns:
            List of prediction results
        """
        if self.model_type == "local":
            return self._infer_batch_local(items)
        else:
            return self._infer_batch_parallel_api(items)

    def _infer_batch_local(self, items: List[Dict]) -> List[Dict]:
        """Batch inference for local Qwen model."""
        results = []

        # Prepare all prompts
        all_messages = []
        for item in items:
            action = item.get("action")
            version = item.get("version")

            # Always use action_triggered templates
            if version.lower() == "v1":
                template = ACTION_TRIGGER_EVAL_TEMPLATE_V1
            elif version.lower() == "v1_cot":
                template = ACTION_TRIGGER_EVAL_TEMPLATE_V1_WITH_COT
            elif version.lower() == "v2":
                template = ACTION_TRIGGER_EVAL_TEMPLATE_V2
            elif version.lower() == "v2_cot":
                template = ACTION_TRIGGER_EVAL_TEMPLATE_V2_WITH_COT
            elif version.lower() == "v3":
                template = ACTION_TRIGGER_EVAL_TEMPLATE_V3
            else:
                raise NotImplementedError("Version Not Found")

            prompt_text = template.format(action=action, safety_principles=SAFETY_PRINCIPLES)
            all_messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image_path"]},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ])

        # Calculate number of batches
        num_batches = (len(all_messages) + self.batch_size - 1) // self.batch_size

        # Process in batches
        with tqdm(total=len(items), desc="Running inference (local)") as pbar:

            for i in range(0, len(all_messages), self.batch_size):
                batch_messages = all_messages[i:i + self.batch_size]
                batch_items = items[i:i + self.batch_size]
                if "think" in self.model_name.lower():
                    max_new_tokens = 8192
                else:
                    max_new_tokens = 512
                try:
                    # Process batch
                    inputs = self.processor.apply_chat_template(
                        batch_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                        padding=True,  # Enable padding for variable-length sequences
                    )
                    inputs = inputs.to(self.model.device)

                    with torch.no_grad():
                        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_texts = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    for j, (item, output_text) in enumerate(zip(batch_items, output_texts)):
                        results.append({
                            "id": item["id"],
                            "image_path": item["image_path"],
                            "raw_output": output_text,
                            "status": "success"
                        })
                        pbar.update(1)

                except Exception as e:
                    print(f"Batch inference error: {e}")
                    # Fallback to single inference for this batch
                    for item in batch_items:
                        output_text = self.infer_single(
                            item["image_path"],
                            item.get("action", ""),
                            item.get("version", "")
                        )
                        results.append({
                            "id": item["id"],
                            "image_path": item["image_path"],
                            "raw_output": output_text,
                            "status": "success"
                        })
                        pbar.update(1)

        return results

    def _infer_batch_parallel_api(self, items: List[Dict]) -> List[Dict]:
        """Parallel API calls for inference."""
        results = [None] * len(items)

        def infer_one(item):
            raw_output = self.infer_single(
                item["image_path"],
                item.get("action", ""),
                item.get("version", ""),
            )
            return {
                "id": item["id"],
                "image_path": item["image_path"],
                "raw_output": raw_output,
                "status": "success" if raw_output is not None else "error"
            }

        # import ipdb; ipdb.set_trace()
        # infer_one(items[0])
        with ThreadPoolExecutor(max_workers=24) as executor:
            future_to_idx = {
                executor.submit(infer_one, item): item["id"]
                for item in items
            }

            with tqdm(total=len(items), desc="Running inference") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        print(f"Error processing item {idx}: {e}")
                        results[idx] = {
                            "id": items[idx]["id"],
                            "image_path": items[idx]["image_path"],
                            "raw_output": str(e),
                            "status": "error"
                        }
                    finally:
                        pbar.update(1)

        return results


def run_inference_phase(agent: SafetyAgent, dataset: List[Dict], version: str, predictions_file: str) -> List[Dict]:
    """
    Run inference phase and save predictions.

    Args:
        agent: SafetyAgent instance
        dataset: Ground truth dataset
        version: Prompt version
        predictions_file: Path to save predictions

    Returns:
        List of valid items with predictions (for evaluation phase)
    """
    valid_items = []
    for i, gt_data in enumerate(dataset):
        if gt_data.get('safety_risk') is None:
            continue
        if gt_data.get("state") == "failed":
            continue

        dr = gt_data['safety_risk']
        image_path = os.path.join("data", dr['edit_image_path']) # .replace('edit_image', 'annotate_image')

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        valid_items.append({
            "id": i,
            "image_path": image_path,
            "action": dr.get("action"),
            "version": version,
            "gt_data": gt_data
        })

    print(f"Running inference on {len(valid_items)} valid samples...")

    results = agent.infer_batch(valid_items)
    
    # Save predictions
    predictions_to_save = []
    for r in results:
        if r["status"] == "success" or r["status"] == "success_fallback":
            predictions_to_save.append({
                "id": r["id"],
                "raw_output": r["raw_output"]
            })

    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_to_save, f, indent=2, ensure_ascii=False)

    print(f"Predictions saved to: {predictions_file}")

    # Merge predictions with GT data for evaluation
    eval_items = []
    for r in results:
        if r["status"] == "success" or r["status"] == "success_fallback":
            valid_item = next(v for v in valid_items if v["id"] == r["id"])
            eval_items.append({
                "id": r["id"],
                "image_path": r["image_path"],
                "raw_output": r["raw_output"],
                "gt_data": valid_item["gt_data"]
            })

    return eval_items
