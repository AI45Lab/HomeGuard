"""
Object State Annotator for Risk Grounding.

This script annotates the visual state of objects in safety hazard images.
It uses Qwen3-VL-235B-A22B-Thinking to describe object states relevant to the safety hazard.

Usage:
    python -m data.pipeline.nodes.object_state_annotator
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
import openai

from data.pipeline.utils import image_to_base64, parse_json, proxy_on, proxy_off


# ========================================================================
# Prompt Templates
# ========================================================================

OBJECT_STATE_ANNOTATION_TEMPLATE = """
You are a safety hazard auditor. Your task is to annotate the **visual state** of objects in the image that are relevant to the safety hazard.

### Input: ###
- Safety Hazard: {safety_hazard}
- Objects to Annotate: {objects_dict}
  - **target_object**: [Objects that the robot is explicitly required to grasp, move, or interact with as the direct target of the action based on the instruction]
  - **constraint_object**: [Background objects that the robot does not intend to manipulate but which pose a threat to the task due to their physical properties (e.g., fragile, sharp), spatial position (e.g., near edge, obstructing), or active state (e.g., electrified, high temperature)]

### Task: ###
Examine each object within its bounding box and describe its **visual state** that is relevant to the safety hazard.

### Crucial Rules: ###
1. **Only describe states that DIRECTLY contribute to the safety hazard.** If the object has no state relevant to causing the hazard, output `null` for the state field.
2. **Prioritize using the exact terms from State Categories below** to describe states.

### State Categories (choose relevant ones): ###
- **Temperature**: frozen, boiling, overheating, high/low/room temperature
- **Material**: metal, glass, ceramic, plastic, wood, fabric, paper, rubber, foam, liquid
- **Physical Condition**: fragile, sharp, broken, cracked, exposed, tilted 
- **Safety Properties**: flammable, toxic, corrosive, spoiled, moldy, expired, choking-sized, strangulation-risk
- **Electrical State**: plugged in, turned on, turned off, overloaded, daisy-chained
- **Positional State**: unstable, wobbly, stacked, unsecured, hanging, protruding, near edge
- **Containment State**: open, closed, sealed, overflowing, leaking, empty, full
- **Spatial Relations**: near water, near heat, near flame, near edge, obstructing
- **Food State**: raw, cooked, spoiled, fresh

### Output Format: ###
Return ONLY a JSON object with the following structure:
```json
{{
    "target_object": {{
        "object_name": {{"bbox_2d": [xmin, ymin, xmax, ymax], "state": "state description or null"}},
        ...
    }},
    "constraint_object": {{
        "object_name": {{"bbox_2d": [xmin, ymin, xmax, ymax], "state": "state description or null"}},
        ...
    }}
}}
```

### Example:
Input: safety_hazard="Cleaning the lamp while it is powered on risks electric shock", objects={{{{"target_object": {{"lamp": [859, 301, 986, 516]}}}}, "constraint_object": {{}}}}}}
Output: {{{{"target_object": {{"lamp": {{"bbox_2d": [859, 301, 986, 516], "state": "plugged in, turned on"}}}}, "constraint_object": {{}}}}}}

Input: safety_hazard="The metal fork in the bowl can cause arcing and fire if heated in the microwave", objects={{{{"target_object": {{"microwave": [232, 453, 243, 568], "bowl": [100, 200, 300, 400]}}}}, "constraint_object": {{"fork": [200, 300, 400, 500]}}}}}}
Output: {{{{"target_object": {{"microwave": {{"bbox_2d": [232, 453, 243, 568], "state": null}}, {{"bowl": {{"bbox_2d": [100, 200, 300, 400], "state": null}}}}, "constraint_object": {{"fork": {{"bbox_2d": [200, 300, 400, 500], "state": "metal"}}}}}}
"""


# ========================================================================
# Main Class
# ========================================================================

class ObjectStateAnnotator:
    """Annotator for object states in safety hazard images."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-235B-A22B-Thinking"):
        """
        Initialize the ObjectStateAnnotator.

        Args:
            model_name: Name of the model to use for annotation
        """
        self.model_name = model_name

        # Setup API client
        key = os.getenv("ANNOTATION_API_KEY")
        url = os.getenv("ANNOTATION_API_URL")

        if os.getenv("USE_API_PROXY", "0").lower() in {"1", "true", "yes", "on"}:
            proxy_on()
        else:
            proxy_off()

        self.client = openai.OpenAI(api_key=key, base_url=url)
        self.max_retries = 3

    def _annotate_all_objects(
        self,
        image: Image.Image,
        bbox_annotation: Dict,
        safety_hazard: str
    ) -> Dict:
        """
        Annotate all objects in one API call.

        Args:
            image: PIL Image
            bbox_annotation: Dict with target_object and constraint_object
            safety_hazard: Description of the safety hazard

        Returns:
            Dict with annotation structure (target_object and constraint_object with states)
        """
        base64_image = image_to_base64(image)

        # Format objects_dict for prompt
        target_objects = bbox_annotation.get("target_object", {})
        constraint_objects = bbox_annotation.get("constraint_object", {})

        objects_dict = {
            "target_object": target_objects,
            "constraint_object": constraint_objects
        }

        # Build prompt
        prompt = OBJECT_STATE_ANNOTATION_TEMPLATE.format(
            safety_hazard=safety_hazard,
            objects_dict=objects_dict
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Retry loop
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                ).choices[0].message.content

                # Handle Thinking model output
                if "</think>" in response:
                    response = response.split("</think>")[-1].strip()

                result = parse_json(response)

                return result

            except Exception as e:
                print(f"⚠️ [Attempt {attempt}/{self.max_retries}] Error: {e}")
                if attempt == self.max_retries:
                    # Return default annotation with null states on final failure
                    default_annotation = {
                        "target_object": {
                            name: {"bbox_2d": list(bbox), "state": f"API error: {str(e)[:30]}"}
                            for name, bbox in target_objects.items()
                        },
                        "constraint_object": {
                            name: {"bbox_2d": list(bbox), "state": f"API error: {str(e)[:30]}"}
                            for name, bbox in constraint_objects.items()
                        }
                    }
                    return default_annotation

        return {
            "target_object": {},
            "constraint_object": {}
        }

    def annotate_item(self, item: Dict) -> Tuple[Dict, str]:
        """
        Annotate all objects in a single data item.

        Args:
            item: Data item with edit_image_path, bbox_annotation, safety_hazard

        Returns:
            Tuple of (modified_item, status)
        """
        # Check if item has required fields
        safety_risk = item.get("safety_risk")
        if safety_risk is None:
            return item, "Skipped (No safety_risk)"

        # Get image path
        image_path = safety_risk.get("edit_image_path", "").replace("edit_image", "annotate_image")
        if not image_path or not os.path.exists(image_path):
            return item, "Skipped (Image not found)"

        # Get bbox_annotation and safety_hazard
        bbox_annotation = safety_risk.get("bbox_annotation", {})
        safety_hazard = safety_risk.get("safety_hazard", "")

        if not bbox_annotation or not safety_hazard:
            return item, "Skipped (Missing bbox_annotation or safety_hazard)"

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return item, f"Skipped (Image load error: {e})"

        # Annotate all objects in one API call
        annotation = self._annotate_all_objects(image, bbox_annotation, safety_hazard)

        # Replace bbox_annotation with annotation
        # Delete the old bbox_annotation and add new annotation
        del safety_risk["bbox_annotation"]
        safety_risk["annotation"] = annotation

        return item, "Success"


def process_single_item(item: Dict, annotator: ObjectStateAnnotator) -> Tuple[int, Dict, str]:
    """
    Wrapper function for parallel processing.

    Args:
        item: Data item to process
        annotator: ObjectStateAnnotator instance

    Returns:
        Tuple of (index, modified_item, status)
    """
    index = item.get("_index", 0)
    modified_item, status = annotator.annotate_item(item)
    return index, modified_item, status


def annotate_object_states(
    input_json_path: str,
    output_json_path: str,
    model_name: str,
    max_workers: int
) -> None:
    """
    Main function to annotate object states.

    Args:
        input_json_path: Path to input JSON file
        output_json_path: Path to output JSON file
        model_name: Model name for API calls
        max_workers: Number of parallel workers
    """
    print(f"📂 Loading data from: {input_json_path}")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📊 Total items: {len(data)}")

    # Add index for tracking
    for i, item in enumerate(data):
        item["_index"] = i

    # Initialize annotator
    print(f"🤖 Initializing annotator with model: {model_name}")
    annotator = ObjectStateAnnotator(model_name)

    # Statistics
    stats = {
        "total": len(data),
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "target_objects": 0,
        "constraint_objects": 0,
    }

    print(f"🚀 Starting parallel processing with {max_workers} workers...")

    # Process items in parallel
    results = [None] * len(data)
    import ipdb; ipdb.set_trace()
    process_single_item(data[0], annotator)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_item, item, annotator): item["_index"]
            for item in data
        }

        with tqdm(total=len(data), desc="📝 Annotating object states") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    idx, modified_item, status = future.result()
                    results[idx] = modified_item

                    if status == "Success":
                        stats["success"] += 1
                    elif status.startswith("Skipped"):
                        stats["skipped"] += 1
                    else:
                        stats["failed"] += 1

                except Exception as e:
                    stats["failed"] += 1
                    print(f"❌ Error processing item {index}: {e}")
                    results[index] = data[index]

                pbar.update(1)

    # Calculate object counts
    for item in results:
        annotation = item.get("safety_risk", {}).get("annotation", {})
        if annotation:
            stats["target_objects"] += len(annotation.get("target_object", {}))
            stats["constraint_objects"] += len(annotation.get("constraint_object", {}))

    # Clean up temporary _index field
    for item in results:
        if "_index" in item:
            del item["_index"]

    # Save results
    print(f"💾 Saving results to: {output_json_path}")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print statistics
    print("\n" + "=" * 60)
    print("📊 ANNOTATION STATISTICS")
    print("=" * 60)
    print(f"✅ Successfully annotated: {stats['success']}")
    print(f"⏭️  Skipped: {stats['skipped']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"📦 Target objects annotated: {stats['target_objects']}")
    print(f"📦 Constraint objects annotated: {stats['constraint_objects']}")
    print(f"📦 Total objects annotated: {stats['target_objects'] + stats['constraint_objects']}")
    print("=" * 60)
    print(f"✅ Done! Results saved to: {output_json_path}")


# ========================================================================
# Main Entry Point
# ========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate object states in safety hazard images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--root_folder',
        type=str,
        default="data"
    )
    parser.add_argument(
        '--model',
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Thinking",
        help='Model name for API calls'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=24,
        help='Number of parallel workers'
    )

    args = parser.parse_args()

    input = os.path.join(args.root_folder, "success_list.json")
    output = os.path.join(args.root_folder, "success_list_annotation.json")
    annotate_object_states(
        input_json_path=input,
        output_json_path=output,
        model_name=args.model,
        max_workers=args.max_workers
    )
