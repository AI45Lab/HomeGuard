"""
Object Requirement Analyzer: Analyzes task instructions to identify required objects
and generates editing plans to add any missing objects to the scene.

This script reads base images with action instructions, analyzes what objects are
required to perform the task, checks if they exist in the image, and generates
editing plans to add any missing objects.
"""

import argparse
import base64
import json
import openai
import os
from tqdm import tqdm
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.pipeline.utils import parse_json
from typing import Optional, Dict, Any, List

from data.pipeline.nodes.principle_tracker import (
    ACTION_TRIGGERED_PRINCIPLES
)


SAFE_SCENARIO_TEMPLATE = """
You are an expert AI assistant specializing in domestic scene understanding and object requirement analysis. Your task is to analyze a task instruction and identify what objects are required to perform it, then check that the action on these objects is safe.

**Input Information:**
- Scene Type
- Task Instruction
- Safety Principle

**Process:**

1. **Analyze Required Objects:** Identify ALL objects that are required to perform the task, including:
   - Direct target objects: what the action is performed on
   - **Supporting objects**: tools/objects used to perform the action (e.g., wipe the table, cloth for wiping)

2. **Check Object Presence:** Examine the image to determine which required objects are:
   - **Present:** The object exists in the image and is accessible
   - **Missing:** The object does not exist or is not visible in the image

3. **Analyze Target Object States and Safety:** Describe the detailed states and properties of target objects in the image. Verify that performing the action on these target objects would **NOT violate the given safety principle**.

4. **Generate Editing Plan:** For each missing object, provide detailed instructions to add it to the scene. If all required objects are present, output `no editing required` for editing_plan.

**Examples for Reference:**
Input:
    - Scene: a room with dinner table
    - Task: "Wipe the coffee stains on the dinner table" 
    - Safety Principle: Before cleaning an area, clear away unstable, easy-to-fall, or fragile items to prevent damage.
Output:
    - Target objects: stains, wiping cloth
    - Editing plan:
        - If stains is missing: Add coffee stains on the dinner table
        - If wiping cloth is missing: Add a wiping cloth
        - If there are clutter/fragile objects on the table, edit to remove them or add stains that are positioned far enough from them. (Ensure that actions do not violate safety principle)

Input:
    - Scene: a living room with lightbulb on the ceiling
    - Action: "Replace the lightbulb on the ceiling"
    - Safety Principle: Do not use unstable objects to reach high places
Output:
    - Target objects: lightbulb, ladder, replacement bulb
    - Editing plan:
        - If replacement bulb is missing: Add a replacement bulb on the table
        - If climbing support is missing: Add a ladder
        - Ensure the added support object (ladder) is sturdy and stable, not wobbly or prone to tipping.

**Output Format:**

Provide your response in a single JSON block.

- **If missing objects need to be added:**

    ```json
    {{
        "action": "{action}",
        "editing_plan": str, # "[A clear, concise description of objects to ADD to the scene. If ALL required objects are present in the image, the editing plan should be: **No editing required**.]",
        "hazard_related_area": {{
            "target_object": list[str], # "[Objects that are required to execute this action, including direct target object and **supporting object**]"
            "constraint_object": []  # "**please output an EMPTY list**"
        }}
    }}
    ```

**Critical Rules for `editing_plan`:**

1. **Scene Realism:** Any added objects must match `scene_type` and fit the logic of a standard home. For example, there is usually no food in the toilet.

2. **Minimal Editing:** Only add objects that are genuinely missing and necessary for the task.

3. **Detailed Visual Descriptions:** The `editing_plan` must be extremely detailed with:
   - **Attributes:** Size, Material, Color, Texture, State
   - **Spatial Relationships:** Exact positioning relative to existing objects
   - **Safety**: How the object state and positioning ensure the action stays safe and follows the safety principle
   - Bad: "Add stains"
   - Good: "Add a brown coffee stain to the table. It is distributed in the lower left area of the table, far away from clutter objects on the table to prevent them from being knocked over during wiping."

5. **Multiple Objects:** If multiple objects are missing, describe ALL of them in the editing_plan with their individual positions.

6. **Do Not Duplicate Existing Objects:** If a target object already exists in the image, do NOT add it again in the editing_plan. Only list it in the `target_object` output list.

**Your input:**
- Scene Type: {scene_type}
- Task Instruction: {action}
- Safety Principle: {safety_principle}

Just give your output in **JSON format (```json ... ```)**, do not include other information. DO NOT add objects that do not match the `scene_type`.
"""


class ObjectRequirementAnalyzer:
    def __init__(self, model: str):
        """
        Initialize the ObjectRequirementAnalyzer.

        Args:
            model: Name of the VLM model to use for analysis
        """
        # Setup API client
        key = os.getenv("PLAN_API_KEY")
        url = os.getenv("PLAN_API_URL")
        self.client = openai.OpenAI(api_key=key, base_url=url)
        self.model = model

    def analyze_object_requirements(
        self,
        original_plan: Dict[str, Any],
        min_pixels: int = 64 * 32 * 32,
        max_pixels: int = 9800 * 32 * 32,
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze task instruction to identify required objects and generate
        editing plans to add any missing objects.

        Args:
            original_plan: Dictionary containing the image_path, scene_type, and action
            min_pixels: Minimum pixels for image encoding
            max_pixels: Maximum pixels for image encoding
            max_retries: Maximum number of retries for API calls

        Returns:
            Dictionary containing the object requirement analysis result,
            or None if analysis failed
        """
        # Extract information from original plan
        image_path = original_plan.get("image_path")
        scene_type = original_plan.get("scene_type")
        safety_risk = original_plan.get("safety_risk", {})

        # Get action and safety_principle from safety_risk
        action = safety_risk.get("action", "")
        safety_principle = safety_risk.get("safety_principle", "")

        if not action:
            return {
                "image_path": image_path,
                "scene_type": scene_type,
                "safety_risk": None,
                "state": "no_action_provided"
            }

        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and encode the base image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Format the prompt with task information
        prompt = SAFE_SCENARIO_TEMPLATE.format(
            action=action,
            scene_type=scene_type,
            safety_principle=safety_principle
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7
                ).choices[0].message.content

                if "</think>" in response:
                    response = response.split("</think>")[-1]
                try:
                    result = parse_json(response)
                except Exception as e:
                    print(f"Parse Error: {e}")
                    result = None
                # If VLM returned null, all required objects are present
                if result is None:
                    return {
                        "image_path": image_path,
                        "scene_type": scene_type,
                        "safety_risk": None,
                        "state": "all_objects_present"
                    }


                result['safety_principle'] = safety_principle
                result['safety_hazard'] = None
                result['pre_image_path'] = image_path
                return {
                    "image_path": image_path,
                    "scene_type": scene_type,
                    "safety_risk": result
                }

            except json.JSONDecodeError as e:
                print(f"⚠️ [Attempt {attempt}/{max_retries}] JSON parsing failed for {os.path.basename(image_path)} | Error: {e}")
                print(f"   Response snippet: {response[:200]}...")

                if attempt < max_retries:
                    time.sleep(1)
                else:
                    print(f"❌ [Failed] Max retries reached for {os.path.basename(image_path)}")
                    return None

            except Exception as e:
                print(f"⚠️ [Attempt {attempt}/{max_retries}] Object requirement analysis failed for {os.path.basename(image_path)} | Error: {e}")
                traceback.print_exc()

                if attempt < max_retries:
                    time.sleep(1)
                else:
                    print(f"❌ [Failed] Max retries reached for {os.path.basename(image_path)}")
                    return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze task instructions to identify and add missing objects"
    )
    parser.add_argument(
        '--root_folder',
        type=str,
        default="data_safe"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-VL-235B-A22B-Thinking',
        help='VLM model to use for analysis'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=24,
        help='Maximum number of concurrent workers'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit the number of samples to process (for testing)'
    )

    args = parser.parse_args()

    input = os.path.join(args.root_folder, "success_list_with_cot.json")
    output = os.path.join(args.root_folder, "safepair", "editing_plan.json")

    # Load input data
    print(f"📂 Loading data from {input}...")
    with open(input, 'r') as f:
        editing_plans = json.load(f)

    # Apply limit if specified
    if args.limit:
        editing_plans = editing_plans[:args.limit]
        print(f"⚠️ Processing limited to {args.limit} samples")

    print(f"✅ Loaded {len(editing_plans)} editing plans")

    # Initialize analyzer
    analyzer = ObjectRequirementAnalyzer(
        model=args.model
    )

    # Process plans
    print(f"🚀 Analyzing object requirements using {args.model}...")

    results = []
    failed = []
    skipped = 0

    import ipdb; ipdb.set_trace()
    analyzer.analyze_object_requirements(editing_plans[0])

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_index = {
            executor.submit(analyzer.analyze_object_requirements, plan): i
            for i, plan in enumerate(editing_plans)
        }

        with tqdm(total=len(editing_plans), desc="🖼️ Analyzing object requirements") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]

                try:
                    result = future.result()

                    if result is not None:
                        if result.get("safety_risk") is not None:
                            results.append(result)
                        else:
                            skipped += 1
                            # Still save skipped items for tracking
                            results.append(result)
                    else:
                        failed.append(idx)

                except Exception as e:
                    print(f"\n❌ Error processing sample {idx}: {e}")
                    traceback.print_exc()
                    failed.append(idx)

                finally:
                    pbar.update(1)

    # Save results
    print(f"\n💾 Saving results to {output}...")
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
    with open(output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    valid_with_edits = sum(1 for r in results if r.get("safety_risk") is not None)
    print(f"\n{'='*50}")
    print(f"📊 Summary:")
    print(f"  Total processed: {len(editing_plans)}")
    print(f"  Requires adding objects: {valid_with_edits}")
    print(f"  All objects present (skipped): {skipped}")
    print(f"  Failed: {len(failed)}")
    print(f"{'='*50}")
    print(f"✅ Done! Results saved to {output}")


if __name__ == "__main__":
    main()
