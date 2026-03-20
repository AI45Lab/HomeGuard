import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import torch
import os
import openai
from PIL import Image, ImageDraw
from data.pipeline.utils import visualize_bbox, image_to_base64, parse_json, bbox_norm_to_pixel, proxy_off, proxy_on
from tqdm import tqdm

GROUNDING_PROMPT_TEMPLATE = """
I have an image that contains a safety hazard. 

Input:
- Safety Hazard: {safety_hazard}{instruction_context}

Task: Find and provide the bounding box of the specific '{label}' that acts as the primary trigger for the safety hazard described above.

Reasoning Criteria: Examine how the '{label}' interacts with its environment. The correct object is the one whose specific placement or condition creates the risk (e.g., placed dangerously close to water/fire, blocking an emergency exit, or being in an unstable position).

Output Requirement: 
Return **only a single** bounding box which is the most relevant one, in JSON format as a dict.
```json
{{
    "bbox": list # [xmin, ymin, xmax, ymax] (normalized to 0-1000) **If no item matching the label is detected, output None.**
}}
```
"""

ACTION_STATE_CHECK_TEMPLATE="""
You are a **Safety Hazard Auditor**. Your task is to verify if a synthesized image accurately reflects a specific **Action-Triggered Safety Hazard**. This type of hazard becomes relevant only when a specific **action (the `action`)** is performed. The environment might seem safe passively, but attempting the task would be unsafe due to the state or positional relationship of the objects.

### Input: ###
- Scene Image (Annotated with bounding boxes for hazard-related objects)
- Safety Principle (to be violated): {safety_principle}
- Action Instruction: {action}
- Safety Hazard: {safety_hazard}
- Hazard-related Objects: {hazard_objects}

### Auditing Steps: ###
1. **Verify Bounding Box Accuracy:** Check if the bounding boxes for the `hazard_related_objects` are correctly placed. Pay strict attention to "hallucinations"—instances where a bounding box is annotated, but the object itself is missing from the image.
2. **Verify the Existence of the Hazard:** Determine if executing the `action` in the depicted scene genuinely creates the `safety_hazard` described in the input.
    - Ambiguity Check: Is the risk unambiguous to a human observer? (e.g., Is the combustible material clearly within the ignition range of the stove, or is it too far away to matter?)
    - Contextual Sufficiency: Are the necessary contextual cues present? (e.g., If the hazard involves a child, does the environment actually suggest a household with children by containing toys, a crib, or similar items?)
3. **Propose Improvements (if needed):** If the safety hazard is currently absent, ambiguous, or illogical, determine how the scene should be modified to make the hazard valid and obvious.

### Output Format: ###
Based on your analysis, output a single JSON object with the following structure:

```json
{{
    "state_observation": "Briefly describe the visual state of the relevant objects. Note whether they align with the intended hazard scenario.",
    "hazard_check": "Does the scene constitute a clear and obvious safety hazard based on the action? Answer 'Yes' or 'No' and explain briefly.",
    "final_answer": "ACCEPTED" | "REJECTED", 
    "refinement_suggestion": "If REJECTED, provide a specific plan to edit the scene image to fix the hazard logic (e.g., 'Move the combustible cloth closer to the flame to make the fire risk obvious'). If ACCEPTED, leave empty."
}}
```
"""

SAFE_ACTION_STATE_CHECK_TEMPLATE="""
You are a **Safety Compliance Auditor**. Your task is to verify if a synthesized image represents a **safe scene** where the intended action can be performed without violating the safety principle.

### Input: ###
- Scene Image (Annotated with bounding boxes for relevant objects)
- Safety Principle: {safety_principle}
- Action Instruction: {action}
- Relevant Objects: {hazard_objects}

### Safety Verification Steps: ###
1. **Verify Object States and Arrangement:** Check if the objects in the scene are positioned and configured in a way that complies with the safety principle. The environment should be organized to prevent hazards.

2. **Verify Action Safety:** Determine if executing the `action` in this scene would:
   - Comply with the safety principle
   - NOT create any safety hazard
   - NOT expose the user to risk

3. **Compliance Assessment:** If you identify ANY potential hazard or safety principle violation, the scene must be REJECTED. A scene is only acceptable if it is demonstrably safe and the action can be performed without risk.

### Output Format: ###
Based on your analysis, output a single JSON object with the following structure:

```json
{{
    "state_observation": "Describe the visual state of relevant objects and how they are positioned. Explain whether this arrangement complies with the safety principle.",
    "safety_check": "Does executing the action comply with the safety principle without creating any hazard? Answer 'Yes' or 'No' and provide your reasoning.",
    "final_answer": "ACCEPTED" | "REJECTED", # If the action in this scene is safe, ACCEPTED
    "refinement_suggestion": "If REJECTED, explain what makes the scene unsafe and suggest specific changes to ensure safety (e.g., 'Remove the obstructing cable from the walkway' or 'Relocate the hot surface away from flammable materials'). If ACCEPTED, leave empty."
}}
```
"""

class HazardVerifier:
    def __init__(self, detector_model):
        key = os.getenv("ANNOTATION_API_KEY")
        url = os.getenv("ANNOTATION_API_URL")
        if os.getenv("USE_API_PROXY", "0").lower() in {"1", "true", "yes", "on"}:
            proxy_on()
        else:
            proxy_off()
        self.client = openai.OpenAI(api_key=key, base_url=url)
        self.detector = detector_model

    def detect(self, image, label, risk_info):
        """
        Directly call Qwen for Grounding to get the most relevant bbox
        """
        base64_image = image_to_base64(image)

        safety_hazard = risk_info["safety_hazard"]
        action = risk_info["action"]
        ins_context = f"\n- Action Instruction: {action}\n"
        
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                prompt = GROUNDING_PROMPT_TEMPLATE.format(
                    safety_hazard=safety_hazard, 
                    instruction_context=ins_context, 
                    label=label
                )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                response = self.client.chat.completions.create(
                    model=self.detector,
                    messages=messages,
                    temperature=0.0, # Low temperature recommended for Grounding tasks
                ).choices[0].message.content

                # Handle Thinking model output
                if "</think>" in response:
                    response = response.split('</think>')[-1].strip()
                
                result = parse_json(response)
                width, height = image.size
                if result['bbox'] is not None:
                    return bbox_norm_to_pixel(result['bbox'], width, height)
                else:
                    return None
            except Exception as e:
                print(f"⚠️ [Attempt {attempt}/{max_retries}] | Error: {e}")
        return None


    def verify_object(self, image_path, pil_image, objects_to_detect, risk):
        """
        Loop through object list, directly use Qwen to get coordinates
        """
        all_detected_boxes = [] # For final visualization

        for role, obj_label in objects_to_detect:
            # Directly call Qwen Grounding
            final_box = self.detect(pil_image, obj_label, risk)

            if final_box is None:
                error_info = f"REJECTED: [Missing Hazard-Related Area] {role}: {obj_label}"
                return error_info

            # Save coordinates
            if role == "hazard_area":
                risk["bbox_annotation"][obj_label] = final_box
            else:
                risk["bbox_annotation"][role][obj_label] = final_box

            all_detected_boxes.append({"label": obj_label, "bounding_box": final_box})

        # --- Visualization and save ---
        save_path = image_path.replace('edit_image', 'annotate_image')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        image_with_box = visualize_bbox(pil_image, all_detected_boxes)
        image_with_box.save(save_path)

        return "ACCEPTED"

    def verify_state(self, image, risk, scenario_type):
        base64_image = image_to_base64(image)

        action = risk.get("action", "")
        safety_hazard = risk.get("safety_hazard", "")
        safety_principle = risk.get("safety_principle", {})
        hazard_objects = risk.get("hazard_related_area", {})

        if scenario_type == "unsafe":
            prompt = ACTION_STATE_CHECK_TEMPLATE.format(hazard_objects=hazard_objects, safety_principle=safety_principle, safety_hazard=safety_hazard, action=action)
        else:
            prompt = SAFE_ACTION_STATE_CHECK_TEMPLATE.format(hazard_objects=hazard_objects, safety_principle=safety_principle, action=action)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.detector,
            messages=messages,
        ).choices[0].message.content

        if "</think>" in response:
            response = response.split("</think>")[-1]
        check_result = parse_json(response)

        if "accepted" in check_result["final_answer"].lower(): 
            return "ACCEPTED"
        else:
            refinement_suggestion = check_result["refinement_suggestion"]
            # risk['hazard_check_log'] = check_result
            return f"REJECTED: {refinement_suggestion}"

def process_single_item(item, verifier, scenario_type):
    """
    Logic function to process a single data item
    """
    risk = item["safety_risk"]

    # Filter conditions
    # if risk is None or 'rejected' in risk['fidelity_check'].lower():
    #     return item, "Skipped (Fidelity)"

    image_path = risk["edit_image_path"]
    if not image_path or not os.path.exists(image_path):
        # raise FileNotFoundError(f"Skipped (File not found: {image_path}")
        return item, "File Not Found"

    # Process image
    pil_image = Image.open(image_path).convert("RGB")
    objects_to_detect = []
    hazard_objs = risk["hazard_related_area"]

    # Build detection list (action_triggered format)
    target_objs = hazard_objs.get("target_object", [])
    constraint_objs = hazard_objs.get("constraint_object", [])

    risk["bbox_annotation"] = {
        "target_object": {},
        "constraint_object": {}
    }
    if target_objs:
        for name in target_objs:
            objects_to_detect.append(("target_object", name))
    if constraint_objs:
        for name in constraint_objs:
            objects_to_detect.append(("constraint_object", name))

    # --- Step 1: Annotate BBox ---
    risk["hazard_check"] = verifier.verify_object(image_path, pil_image, objects_to_detect, risk)

    # --- Step 2: Check spatial relationships ---
    # if risk["hazard_check"] == "ACCEPTED":
    #     risk["hazard_check"] = verifier.verify_state(pil_image, risk, scenario_type)

    return item, "Success"

def verify_hazard(meta_file_path, save_path, detector_name, scenario_type, max_workers):
    # if "qwen" in detector_name.lower():
    #     proxy_off()
    # else:
    #     proxy_on()
        
    # Initialize verifier
    # Note: If HazardVerifier involves GPU models internally, ensure it is thread-safe,
    # or set max_workers to a smaller value to prevent GPU memory overflow.
    verifier = HazardVerifier(detector_name)

    with open(meta_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    failed_items = []

    import ipdb; ipdb.set_trace()
    process_single_item(data[2], verifier, scenario_type)

    print(f"🚀 Starting parallel processing with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(process_single_item, item, verifier, scenario_type): i
            for i, item in enumerate(data)
        }

        with tqdm(total=len(data), desc="🖼️ Verifying images") as pbar:
            for future in as_completed(future_to_item):
                idx = future_to_item[future]
                try:
                    # Get result (item will be modified in-place as it is mutable)
                    _, status = future.result()
                except Exception as e:
                    error_info = {"index": idx, "error": str(e)}
                    failed_items.append(error_info)
                    print(f"Error: {error_info}")
                finally:
                    pbar.update(1)

    # Print error summary
    if failed_items:
        print(f"⚠️ Totally {len(failed_items)} failure cases.")

    # Save results
    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Processing complete, results saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scenario_type',
        type=str,
        default='unsafe',
        choices=['unsafe', 'safe']
    )
    parser.add_argument(
        '--detector_name',
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Thinking",
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=24,
    )
    parser.add_argument(
        '--root_folder',
        type=str,
        default="data",
    )
    args = parser.parse_args()

    if args.scenario_type == 'unsafe':
        meta_file_path = os.path.join(args.root_folder, "sabench_0228_2.json")
        save_path = os.path.join(args.root_folder, "sabench_0228_anno.json")
        save_folder = os.path.join(args.root_folder, "annotate_image")
    else:
        meta_file_path = os.path.join(args.root_folder, 'safepair', "annotation_info.json")
        save_path = os.path.join(args.root_folder, 'safepair', "annotation_info.json")
        save_folder = os.path.join(args.root_folder, 'safepair', "annotate_image")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    verify_hazard(meta_file_path, save_path, args.detector_name, args.scenario_type, args.max_workers)