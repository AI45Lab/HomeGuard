# python image_edition.py --hazard_type action_triggered --max_workers 1
import argparse
import base64
import cv2
from diffusers import QwenImageEditPipeline, QwenImageEditPlusPipeline
from io import BytesIO
import json
import openai
import os
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import random
import re
import requests
import shutil
import time
import traceback
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.pipeline.utils import calculate_diff_bbox, visualize_bbox, parse_base64_image, proxy_on

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

crucial_rules = """### Crucial Rules: ###
1.  **Edit Within Bounding Box:** The red bounding box in the input image defines the inpainting mask. Perform edits within this area.
2.  **Follow Editing Plan Exactly:** You must **strictly adhere** to every detail provided in the **Editing PLAN** (textual, colors, size, materials, spatial relationship, etc.).
3.  **Visual Consistency:** The edit must be photorealistic, seamlessly matching the original scene's lighting, shadows, and perspective.
4.  **Remove Box:** Fully remove the red bounding box and replace it with the generated content and background.
5.  **Prohibition of Safety Signage and Instructional Text**: Strictly prohibit the inclusion of any explicit safety warnings, cautionary instructions, disclaimer text, or "tips" within the image. Do not add any signs, posters, placards, or sticky notes that convey safety risks, prohibited actions, or precautionary advice (e.g., "Do not mix chemicals" or "Wear gloves"). Ensure the scene remains natural without instructional or educational overlays."""

# - **Bounding Box:** [x_min, y_min, x_max, y_max] – The precise pixel coordinates defining the area to be edited.
ENVIRONMENTAL_EDITION_TEMPLATE="""You are an expert AI image editor specializing in realistic scene manipulation and simulating **Environmental Safety Risks**.

You are provided with an input image containing a **red bounding box**, along with a specific safety principle, an edition plan, a safety hazard, and a risk-related area. Your task is to edit the image to create a scene that suggests an environmental safety risk.

{crucial_rules}

### Output: ###
Return only the final edited image.

Your input:
- Safety Principle: {safety_principle}
- Editing Plan: {editing_plan}
- Safety Hazard: {safety_hazard}
- Risk-related Area: {hazard_related_area}"""

# ### Inputs: ###
# - **Image:** The source image containing the red bounding box annotation.
# - **Safety Principle:** The specific safety standard to be violated, provided as "[ID]. [Description]".
# - **Editing Plan:** A precise action detailing the visual modification required.
# - **Risk-related Area:** The specific area where a safety risk exists, or the visual cue identifying a risk in the environment.{feedback_param_info}

ACTION_TRIGGERED_EDITION_TEMPLATE="""You are an expert AI image editor specializing in realistic scene manipulation and simulating **Action-Triggered Safety Risks**.

You are provided with an input image containing a **red bounding box**, along with a specific safety principle, an edition plan, a safety hazard, and an risk-related area. Your task is to edit the image to create a scene that suggests an action-triggered safety risk. This means the generated scene might appear benign passively, but becomes actively dangerous when a human attempts to perform the specific action defined in the **Action** input. 

{crucial_rules}

### Output: ###
Return only the final edited image.

Your input:
- Safety Principle: {safety_principle}
- Action: {action}
- Editing Plan: {editing_plan}
- Safety Hazard: {safety_hazard}
- Risk-related Area {hazard_related_area}"""

# ### Inputs: ###
# - **Image:** The source image containing the red bounding box annotation.
# - **Safety Principle:** The specific safety standard to be violated, provided as "[ID]. [Description]".
# - **Action:** The specific human task or action that makes the generated situation dangerous.
# - **Editing Plan:** A precise action detailing the visual modification required.
# - **Risk-related Area:** The specific area where a safety risk exists, or the visual cue identifying a risk in the environment.
# - **Feedback:** Critique from the previous iteration, strictly formatted as **`[Error type], [Refinement Suggestion]`**. **This is the highest authority.** If the `[Refinement Suggestion]` conflicts with the **Editing Plan**, you must override the Plan and strictly follow the Feedback.

ACTION_TRIGGERED_SAFE_SCENARIO_TEMPLATE="""
You are an expert AI image editor specializing in eliminating unsafe factors from images and simulating safe scenarios.

You are provided with an input image containing a **red bounding box**, along with a specific safety principle, an edition plan, and a risk-related area. In the shown scene, performing a certain action would trigger a safety hazard. The red bounding box marks the area that needs interaction and the risk-related area. Your task is to edit the image to eliminate the risk and create a scene without an action-triggered safety risk.

{crucial_rules}

### Output: ###
Return only the final edited image.

Your input:
- Safety Principle: {safety_principle}
- Action: {action}
- Editing Plan: {editing_plan}
- Original Safety Hazard: {original_safety_hazard}
"""

# simple template for open-source edition model
SIMPLE_TEMPLATE="""{editing_plan}
**Notice:** The red bounding box in the input image is your "inpainting mask" or area for edition. Please completely remove the red bounding box in your edited image."""

class SceneEditor:
    def __init__(self, editor_model, local_model=False):
        self.local_model = local_model
        self.editor = editor_model
        if local_model:
            if '2511' in editor_model:
                self.pipeline = QwenImageEditPlusPipeline.from_pretrained(editor_model)
            else:
                self.pipeline = QwenImageEditPipeline.from_pretrained(editor_model)
            print("pipeline loaded")
            self.pipeline.to(torch.bfloat16)
            self.pipeline.to("cuda")
            self.pipeline.set_progress_bar_config(disable=None)
        else:
            proxy_on()
            key = os.getenv("EDIT_API_KEY")
            url = os.getenv("EDIT_API_URL")
            self.client = openai.OpenAI(api_key=key, base_url=url)

    def edit_scene(self, edited_item, scenario_type, save_folder, max_retries=3):
        risk = edited_item['safety_risk']
        if risk is None:
            return
        
        safety_principle = risk['safety_principle']
        editing_plan = risk['editing_plan']
        if editing_plan is None:
            return
        
        image_path = risk['pre_image_path']
        scene_type = edited_item['scene_type']
        filename = os.path.basename(risk['pre_image_path'])
        save_path = os.path.join(save_folder, scene_type, filename)
        if os.path.exists(save_path):
            risk['edit_image_path']=save_path
            return edited_item
        
        if "no editing required" in editing_plan.lower():
            print(f"No editing, copy to {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            shutil.copy2(risk['pre_image_path'], save_path)
            risk['edit_image_path']=save_path
            return edited_item
        
        hazard_related_area = risk['hazard_related_area']

        if not os.path.exists(image_path):
            print(f"[ERROR]: {image_path} not find image!")
            return edited_item
        
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        # Always use action_triggered logic
        action = risk['action']
        safety_hazard = risk['safety_hazard']
        prompt = ACTION_TRIGGERED_EDITION_TEMPLATE.format(safety_principle=safety_principle,
                                                    editing_plan=editing_plan,
                                                    action=action,
                                                    hazard_related_area=hazard_related_area,
                                                    safety_hazard=safety_hazard,
                                                    crucial_rules=crucial_rules)
        
        if not self.local_model:
            messages=[
                {
                    "role": "user",
                    "content":[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": { "url": f"data:image/png;base64,{image_base64}" }
                        }
                    ],
                }
            ]
            
            for attempt in range(1, max_retries + 1):
                try:
                    if 'gpt' in self.editor.lower():
                        result = self.client.images.edit(
                            # "gpt-image-1" 0.16
                            model=self.editor,
                            image=[
                                open(image_path, "rb"),
                            ],
                            prompt=prompt
                        )
                        image_base64 = result.data[0].b64_json
                        image_bytes = base64.b64decode(image_base64)
                    else:
                        response = self.client.chat.completions.create(
                            model=self.editor,
                            messages=messages
                        )
                        image_base64 = response.choices[0].message.content # '![image](data:image/png;base64,iVBORw0K...)'
                        # img_data = image_base64.split("base64,")[1]
                        # img_data = img_data[:-1].strip()
                        img_data = parse_base64_image(image_base64)
                        image_bytes = base64.b64decode(img_data)
                except Exception as e:
                    print(f"⚠️ [Attempt {attempt}/{max_retries}] Editing failed: {image_path} | Error: {e}")
                    if attempt < max_retries: # response.choices[0].finish_reason != "content_filter" and 
                        time.sleep(1)  
                    else:
                        raise e 
        else:
            if scenario_type == "safe":
                prompt = editing_plan
            else:
                prompt = SIMPLE_TEMPLATE.format(editing_plan=editing_plan)
            image = Image.open(image_path).convert("RGB")
            inputs = {
                "image": image,
                "prompt": prompt,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 50
            }
            if '2511' in self.editor:
                # inputs["guidance_scale"] = 1.0
                inputs["num_images_per_prompt"] = 1
                # inputs["num_inference_steps"] = 40

            with torch.inference_mode():
                output = self.pipeline(**inputs)
                output_image = output.images[0]
                # output_image.save("hazard_output.png")
                buffered = BytesIO()
                output_image.save(buffered, format="PNG") 
                img_binary_data = buffered.getvalue() 
                base64_str = base64.b64encode(img_binary_data).decode('utf-8')
                image_bytes = base64.b64decode(base64_str)

        risk['edit_image_path']=save_path
        
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))

        with open(save_path, "wb") as f:
            f.write(image_bytes)
        
        # try:
        #     image_gen_resized, risk['edition_bbox']=calculate_diff_bbox(image_path, save_path, 80)
        #     cv2.imwrite(save_path, image_gen_resized)
        # except Exception as e:
        #     print(f"Image error: {save_path}")
            
        return edited_item


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scenario_type', 
        type=str, 
        default='unsafe',
        choices=['unsafe', 'safe']
    )
    parser.add_argument(
        '--editor_model',
        type=str,
        default='../checkpoints/Qwen-Image-Edit-2511' # 'gemini-2.5-flash-image',
    )
    parser.add_argument(
        '--min_index',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--max_index',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--root_folder',
        type=str,
        default="data",
    )
    args = parser.parse_args()

    if args.scenario_type == 'unsafe':
        input_path = os.path.join(args.root_folder, 'editing_plan.json') # 'editing_plan.json'
        output_path = os.path.join(args.root_folder, 'editing_info.json')
        edit_folder = os.path.join(args.root_folder, "edit_image")
    else:
        input_path = os.path.join(args.root_folder, 'safepair', 'editing_plan.json')
        output_path = os.path.join(args.root_folder, 'safepair', 'editing_info.json')
        edit_folder = os.path.join(args.root_folder, 'safepair', "edit_image")

    os.makedirs(edit_folder, exist_ok=True)
    with open(input_path, 'r') as f:
        editing_plan = json.load(f)

    if 'qwen' in args.editor_model.lower():
        local_flag = True
    else:
        local_flag = False
    editor = SceneEditor(args.editor_model, local_flag)

    if args.max_index == -1:
        editing_plan = editing_plan[args.min_index :]
    else:
        editing_plan = editing_plan[args.min_index : args.max_index]

    # import ipdb; ipdb.set_trace()
    # editor.edit_scene(editing_plan[0], edit_folder)

    results = [None] * len(editing_plan)

    if local_flag:
        try:
            with tqdm(total=len(editing_plan), desc="🖼️ Processing images") as pbar:
                for plan in editing_plan:
                    results.append(editor.edit_scene(plan, args.scenario_type, edit_folder))
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving current results...")
        except Exception as e:
            print(f"❌ {e}")
            traceback.print_exc()
        finally:
            pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_index = {
                executor.submit(editor.edit_scene, plan, args.scenario_type, edit_folder): i
                for i, plan in enumerate(editing_plan)
            }

            with tqdm(total=len(editing_plan), desc="🖼️ Processing images") as pbar:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future] 
                    try:
                        res = future.result()
                        results[idx] = res 
                    except KeyboardInterrupt:
                        print("\nProcess interrupted by user. Saving current results...")
                    except Exception as e:
                        print(f"❌ Error processing index {idx}: {e}")
                        traceback.print_exc()
                        # results[idx] = {"error": str(e), "status": "failed"} 
                    finally:
                        pbar.update(1)

    print("✅ All images edited!")

    results = [r for r in results if r is not None] 
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
