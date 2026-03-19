#!/usr/bin/env python3
"""
Item Replacement for Risk Grounding Data Augmentation

This script performs two main tasks:
1. Generate diverse object lists for each safety principle using Gemini-2.5-pro
2. Replace objects in editing plans using Qwen3-VL-235B for validation

Usage:
    # Generate object lists for all principles
    python item_replacement.py --mode generate_objects

    # Replace objects in editing plans
    python item_replacement.py --mode replace --input editing_plan.json --output aug_editing_plan.json
"""

import argparse
import json
import os
import re
import base64
import time
import sys
import traceback
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

# Import principles from principle_tracker
from data.pipeline.nodes.principle_tracker import ACTION_TRIGGERED_PRINCIPLES
from data.pipeline.utils import parse_json, proxy_off, proxy_on

# ========================================================================
# Prompt Templates
# ========================================================================

OBJECT_LIST_GENERATION_TEMPLATE = """
You are helping generate diverse object variations for a safety principle training dataset.

**Safety Principle:**
{principle_id}.{principle_title}: {principle_description}

{principle_examples_section}

**Your Task:**
Generate a comprehensive list of diverse, specific objects that violate this safety principle. Focus on creating objects with different attributes (material, size, color, shape, state) to maximize data diversity.

**Important Guidelines:**

1. **Quantity**: Generate AS MANY objects as possible (aim for 100+ different variations)

2. **Diversity in Attributes:**
   - **Material**: metal, glass, ceramic, plastic, wood, paper, fabric, etc.
   - **Size & Shape**: bowl, plate, cup, tray, container, utensil, tool, etc.
   - **Color & Appearance**: white, red, blue, transparent, opaque, patterned, etc.
   - **State**: new, old, cracked, chipped, dirty, rusty, stained, etc.
   - **Specific Features**: with handle, lid, spout, pattern, brand label, etc.

3. **Scene Consistency**: All objects must be common household items that would realistically appear in the scene type (kitchen, bathroom, living room, etc.)

4. **Specificity**: Each object must be specific and descriptive
   - ❌ Bad: "a bowl"
   - ✅ Good: "a stainless steel mixing bowl"

**Example Format:**
Input Safety Principle: 
3. Material Safety (Microwave/Oven): Ensure only suitable materials (non-metal for microwaves, oven-safe containers for ovens) are placed inside devices.

Output:
```json
{{
  "principle_id": "3",
  "principle_title": "Material Safety (Microwave/Oven)",
  "objects": {{
    "unsuitable object for microwave": ["stainless steel mixing bowl", "aluminum food container with crimped edges", "metal dinner fork with four tines", "metal soup spoon", "metal butter knife with serrated edge", ... ], 
    "unsuitable object for oven": [...]
  }}
}}
```

Input Safety Principle: 
1. Flammable Items Near Heat: Ensure flammable materials and heat sources are kept far apart.

Output:
```json
{{
  "principle_id": "1",
  "principle_title": "Flammable Items Near Heat",
  "objects": {{
    "Flammable Items": ["a stack of paper napkins", "a large cardboard pizza box with grease stains", ... ], 
    "Heat": ["gas stove", "electric warmer", ...]
  }}
}}
```

**Output Format:**
Return ONLY a JSON object in the following format:
```json
{{
  "principle_id": "{principle_id}",
  "principle_title": "{principle_title}",
  "objects": {{
    "object_category": [ ... (continue for 100+ variations)],
    ...
  }}
}}
```

Generate as many diverse objects as possible. Do not limit yourself to a fixed number - the more variations, the better for training data diversity.
"""


# ========================================================================
# Unified CoT Template for Item Replacement
# ========================================================================

UNIFIED_ITEM_REPLACEMENT_COT_TEMPLATE = """
You are a Safety Hazard Data Augmentation Specialist. Your task is to perform intelligent object replacement in safety hazard scenes while maintaining logical consistency and realism.

**Context:**
- Original Action Instruction: "{action}"
- Original Editing Plan: "{editing_plan}"
- Safety Principle: {safety_principle}
- Scene Type: {scene_type}
- Safety Hazard: {safety_hazard}
- Hazard Related Area: {hazard_related_area}

**Replacement Task:**
- Replacement Object: "{replacement_object}" (from category: {replacement_category})

---

## Your Task (Chain of Thought):

### Step 1: Select Object to Replace
From the hazard_related_area, select ONE object to replace with "{replacement_object}".

**Selection Criteria:**
- The object should be semantically similar to or related to the replacement category ({replacement_category})
- The object should be central to creating the safety hazard
- After replacement, the safety hazard should still be valid and logically consistent

**Example:**
- If replacement_category is "unsuitable object for microwave" and replacement_object is "metal spoon"
- You should select a metal object from hazard_related_area (e.g., "metal bowl", "metal fork") to replace
- DO NOT select unrelated objects like "microwave", "ceramic bowl", etc.

### Step 2: Analyze Replacement Feasibility
Analyze whether replacing the selected object with "{replacement_object}" is valid:

**2.1 Scene Compatibility Check:**
- Is "{replacement_object}" plausible in {scene_type}?
- Examples of valid: kitchen -> fork, bowl, tray; bathroom -> toiletries, towels
- Examples of invalid: kitchen -> bathroom items, electric warmer; bathroom -> power tools

**2.2 Action Plausibility Check:**
- Keep the action instruction realistic! Don't change "Heat food in bowl" to "Heat the metal spoon"
- Instead, modify the editing_plan to accommodate the new object while keeping the instruction realistic
- Example: If replacing "metal bowl" with "metal spoon", keep instruction as "Heat food in bowl" but change editing_plan to "Place a ceramic bowl containing food with a metal spoon inside in the microwave"

**2.3 Hazard Logic Check:**
- Does "{replacement_object}" still violate the safety principle?
- The hazard must remain logically consistent

### Step 3: Generate Revised Content
Based on your analysis, generate:
1. **Revised Action Instruction**: Keep original instruction if possible, only modify if necessary for realism
2. **Revised Editing Plan**: Incorporate the replacement object with visual details. May need to add container objects to maintain action plausibility.
3. **Updated Hazard Related Area**: Update target_object and constraint_object lists appropriately

**Example Transformation:**
- Selected object: "metal bowl", replacement: "metal soup spoon"
- Original: action="Heat food in bowl"
- Revised: action="Heat food in bowl" (unchanged - realistic action)
- Revised Plan: "Place a white ceramic bowl containing pasta with a stainless steel soup spoon resting inside in the microwave. The metal spoon will cause arcing when microwave operates."
- Updated hazard_related_area:
  - target_object: ["microwave", "ceramic bowl"]
  - constraint_object: ["stainless steel soup spoon"]

---

## Output Format:

Return ONLY a JSON object:
```json
{{
  "step1_object_selection": {{
    "selected_object": "exact object name from hazard_related_area",
    "reason": "why this object was selected for replacement"
  }},
  "step2_validation": {{
    "scene_compatible": true/false,
    "action_plausible": true/false,
    "hazard_valid": true/false,
    "plan_consistent": true/false
  }},
  "step3_revised_content": {{
    "instruction": "revised instruction (keep original if possible)",
    "editing_plan": "revised editing plan with visual details",
    "safety_hazard": "revised safety hazard",
    "hazard_related_area": {{
      "target_object": ["..."],
      "constraint_object": ["..."]
    }}
  }},
  "final_reasoning": "Brief summary of the entire transformation",
  "final_answer": "ACCEPT" | "REJECT"
}}
```

All four validation checks must be true for ACCEPT.
"""

# ========================================================================
# Object List Generator
# ========================================================================

class ObjectListGenerator:
    """Generate diverse object lists for safety principles"""

    def __init__(self, model, output_dir: str = "data/item_lists"):
        self.api_key = os.getenv("AUG_API_KEY")
        self.api_url = os.getenv("AUG_API_URL")
        self.augmenter = model
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_url)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_for_principle(self, principle_id: int, principle_info: Dict,
                              max_retries: int = 3) -> Optional[Dict]:
        """
        Generate object list for a single safety principle

        Args:
            principle_id: Principle identifier (integer)
            principle_info: Principle info dict with title, description, examples
            max_retries: Maximum API call retries

        Returns:
            Dictionary with principle info and object list, or None if failed
        """
        # Build prompt
        examples_section = ""
        if principle_info.get("examples"):
            # Clean up examples section
            examples_text = principle_info["examples"].strip()
            examples_section = f"\n**Examples:**\n{examples_text}"
        prompt = OBJECT_LIST_GENERATION_TEMPLATE.format(
            principle_id=principle_id,
            principle_title=principle_info["title"],
            principle_description=principle_info["description"],
            principle_examples_section=examples_section
        )

        messages = [{"role": "user", "content": prompt}]

        # Call API with retries
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Generating object list for principle {principle_id} (attempt {attempt}/{max_retries})...")
                response = self.client.chat.completions.create(
                    model=self.augmenter,
                    messages=messages,
                    temperature=0.7
                ).choices[0].message.content

                # Parse JSON
                obj_list = parse_json(response)

                # Validate format
                if not isinstance(obj_list.get("objects"), dict):
                    print(f"⚠ Invalid format: 'objects' should be a dict, got {type(obj_list.get('objects'))}")
                    if attempt < max_retries:
                        time.sleep(2)
                        continue

                obj_dict = obj_list["objects"]
                total_objects = sum(len(items) for items in obj_dict.values())

                if total_objects >= 50:  # At least 50 objects total across all categories
                    print(f"✓ Generated {total_objects} objects across {len(obj_dict)} categories for principle {principle_id}")
                    return obj_list
                else:
                    print(f"⚠ Insufficient objects generated: {total_objects}")

            except Exception as e:
                print(f"⚠ Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    print(f"❌ Failed to generate object list for principle {principle_id}")
                    return None

    def save_object_list(self, obj_list: Dict):
        """Save object list to file"""
        self.output_dir.mkdir(exist_ok=True)

        output_file = self.output_dir / f"principle_{obj_list['principle_id']}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(obj_list, f, indent=2, ensure_ascii=False)

        print(f"Saved object list to {output_file}")
        return output_file

    def generate_all_principles(self):
        """
        Generate object lists for all principles.

        Returns:
            List of successfully generated object lists
        """
        principles = ACTION_TRIGGERED_PRINCIPLES

        print(f"Generating object lists for {len(principles)} principles (action_triggered)...")

        results = []
        failed_principles = []

        for principle_id, principle_info in tqdm(principles.items(), desc="Generating object lists"):
            obj_list = self.generate_for_principle(principle_id, principle_info)
            if obj_list:
                self.save_object_list(obj_list)
                results.append(obj_list)
            else:
                failed_principles.append(principle_id)

        print(f"\n✓ Generated {len(results)}/{len(principles)} object lists successfully")
        if failed_principles:
            print(f"✗ Failed principles: {failed_principles}")

        return results


# ========================================================================
# Item Replacer
# ========================================================================

class ItemReplacer:
    """Replace objects in editing plans with validation"""

    def __init__(self, object_lists_dir: str = "data/item_lists",
                 model: str = "Qwen/Qwen3-VL-235B-A22B-Thinking"):
        self.object_lists_dir = Path(object_lists_dir)

        # Initialize API clients
        check_api_key = os.getenv("REPLACE_API_KEY")
        check_api_url = os.getenv("REPLACE_API_URL")
        if "qwen" in model.lower():
            proxy_off()
        else:
            proxy_on()
        self.client = openai.OpenAI(api_key=check_api_key, base_url=check_api_url)
        self.model = model

        # Load all object lists (nested format: {category: [items]})
        self.object_lists = {}  # {principle_id: {category: [items]}}
        self.flattened_object_lists = {}  # {principle_id: [(category, object), ...]}
        self.current_indices = {}  # {principle_id: global_index}
        self._load_all_object_lists()

    def _load_all_object_lists(self):
        """Load all object lists from directory (nested dict format)"""
        print(f"Loading object lists from {self.object_lists_dir}...")

        for json_file in self.object_lists_dir.rglob("principle_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    principle_id = str(data['principle_id'])
                    objects = data.get('objects', {})

                    # Validate nested dict format
                    if not isinstance(objects, dict):
                        print(f"  ⚠ Invalid format in {json_file}: 'objects' should be a dict")
                        continue

                    # Store nested dict
                    self.object_lists[principle_id] = objects

                    # Create flattened list: [(category, object), ...]
                    flattened = []
                    for category, items in objects.items():
                        for item in items:
                            flattened.append((category, item))

                    self.flattened_object_lists[principle_id] = flattened
                    self.current_indices[principle_id] = 0

                    # Count total objects
                    total_objects = len(flattened)
                    num_categories = len(objects)
                    print(f"  Loaded principle {principle_id}: {total_objects} objects in {num_categories} categories")
            except Exception as e:
                print(f"  ⚠ Failed to load {json_file}: {e}")

        print(f"Loaded {len(self.object_lists)} principle object lists")

    def _extract_principle_id(self, editing_plan: str, safety_principle: str) -> Optional[str]:
        """Extract principle ID from safety principle text"""
        # Try to extract principle number from safety_principle
        match = re.search(r'(\d+)\.\s*', safety_principle)
        if match:
            return match.group(1)

        # Try to match by title
        for pid, info in ACTION_TRIGGERED_PRINCIPLES.items():
            if info['title'] in safety_principle:
                return pid

        return None

    def _extract_objects_from_plan(self, risk_data: Dict) -> List[str]:
        """Extract objects mentioned in hazard_related_area"""
        objects = []
        hazard_area = risk_data.get("hazard_related_area", {})

        # Action triggered format
        for category in ["target_object", "constraint_object"]:
            for obj in hazard_area.get(category, []):
                objects.append(obj)

        return objects

    def _get_next_replacement_object(self, principle_id: str) -> Optional[Tuple[str, str]]:
        """
        Get the next replacement object from the flattened list (cyclic).

        Args:
            principle_id: Principle ID

        Returns:
            Tuple of (category, object) or None if failed
        """
        if principle_id not in self.flattened_object_lists:
            return None

        flattened = self.flattened_object_lists[principle_id]
        if not flattened:
            return None

        # Get current index and update for next call
        current_idx = self.current_indices[principle_id]
        category, obj = flattened[current_idx % len(flattened)]
        self.current_indices[principle_id] = (current_idx + 1) % len(flattened)

        return category, obj

    def _unified_replacement_cot(self, editing_plan: str, safety_principle: str,
                                  action: str, hazard_related_area: Any,
                                  safety_hazard: str,
                                  scene_type: str, principle_id: str,
                                  replacement_object: str,
                                  replacement_category: str) -> Optional[Dict]:
        """
        Unified CoT method that selects object to replace and generates revised content.

        Args:
            editing_plan: Original editing plan
            safety_principle: Safety principle description
            action: User instruction
            hazard_related_area: hazard_related_area data
            scene_type: Scene type
            principle_id: Principle ID
            replacement_object: Replacement object from the list
            replacement_category: Category of the replacement object

        Returns:
            Dict with the complete CoT result or None if failed
        """
        hazard_area_str = json.dumps(hazard_related_area, ensure_ascii=False)

        prompt = UNIFIED_ITEM_REPLACEMENT_COT_TEMPLATE.format(
            action=action,
            editing_plan=editing_plan,
            safety_principle=safety_principle,
            scene_type=scene_type,
            hazard_related_area=hazard_area_str,
            safety_hazard=safety_hazard,
            replacement_object=replacement_object,
            replacement_category=replacement_category
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0
            ).choices[0].message.content

            if "</think>" in response:
                response = response.split("</think>")[-1]
            result = parse_json(response)
            return result
        except Exception as e:
            print(f"  ⚠ Unified CoT failed: {e}")
            return None

    def _update_hazard_related_area(self, risk_data: Dict, original_object: str,
                                    replacement_object: str) -> Dict:
        """Update hazard_related_area with replacement object"""
        hazard_area = risk_data.get("hazard_related_area", {})

        # Action triggered format
        for category in ["target_object", "constraint_object"]:
            objects = hazard_area.get(category, [])
            for i, obj in enumerate(objects):
                if original_object.lower() in obj.lower() or obj.lower() in original_object.lower():
                    objects[i] = replacement_object

        return risk_data

    def replace_item_in_sample(self, sample: Dict) -> Dict:
        """
        Replace items in a single editing plan sample using unified CoT approach.

        Args:
            sample: A single sample from editing_plan.json

        Returns:
            Augmented sample with replacement metadata
        """
        risk_data = sample.get("safety_risk", {})
        if not risk_data:
            return sample

        original_plan = risk_data.get("editing_plan", "")
        safety_principle = risk_data.get("safety_principle", "")
        action = risk_data.get("action", "")
        scene_type = sample.get("scene_type", "unknown")
        hazard_related_area = risk_data.get("hazard_related_area", {})
        safety_hazard = risk_data.get("safety_hazard", "")

        # Extract principle ID
        principle_id = self._extract_principle_id(original_plan, safety_principle)
        if not principle_id or principle_id not in self.object_lists:
            sample.setdefault("_replacement_meta", {})["skipped"] = "No object list available"
            return sample

        # Get next replacement object from flattened list
        replacement_result = self._get_next_replacement_object(principle_id)
        if replacement_result is None:
            sample.setdefault("_replacement_meta", {})["skipped"] = "Failed to get replacement object"
            return sample

        replacement_category, replacement_object = replacement_result

        # Initialize replacement metadata
        replacement_meta = {
            "principle_id": principle_id,
            "replacement_object": replacement_object,
            "replacement_category": replacement_category
        }

        # Use unified CoT approach - single VLM call
        cot_result = self._unified_replacement_cot(
            original_plan, safety_principle, action,
            hazard_related_area, safety_hazard, scene_type, principle_id,
            replacement_object, replacement_category
        )

        if not cot_result:
            sample.setdefault("_replacement_meta", {})["skipped"] = "Unified CoT failed"
            return sample

        # Extract results from CoT
        step1 = cot_result.get("step1_object_selection", {})
        step2 = cot_result.get("step2_validation", {})
        step3 = cot_result.get("step3_revised_content", {})
        final_answer = cot_result.get("final_answer", "REJECT")

        # Get the selected object from VLM's selection
        original_object = step1.get("selected_object", "")

        # Store CoT reasoning in metadata
        replacement_meta.update({
            "original_object": original_object,
            "step1_object_selection": step1,
            "step2_validation": step2,
            "final_reasoning": cot_result.get("final_reasoning", "")
        })

        if final_answer == "ACCEPT" and all(step2.get(k, False) for k in ["scene_compatible", "action_plausible", "hazard_valid", "plan_consistent"]):
            # Apply revised content from CoT
            revised_action = step3.get("action", action)
            revised_editing_plan = step3.get("editing_plan", original_plan)
            revised_hazard_area = step3.get("hazard_related_area", hazard_related_area)
            revised_safety_hazard = step3.get("safety_hazard", safety_hazard)

            # Update risk data with revised content
            risk_data["action"] = revised_action
            risk_data["editing_plan"] = revised_editing_plan
            risk_data["hazard_related_area"] = revised_hazard_area
            risk_data["safety_hazard"] = revised_safety_hazard

            replacement_meta["replaced"] = True
            replacement_meta["original_plan"] = original_plan
            replacement_meta["original_action"] = action
            # print(f"  ✓ Replaced '{original_object}' with '{replacement_object}' (category: {replacement_category})")
        else:
            replacement_meta["skipped"] = cot_result.get("final_reasoning", "Replacement rejected by CoT")
            print(f"  ✗ Replacement rejected: {cot_result.get('final_reasoning', 'Unknown')}")

        sample["_replacement_meta"] = replacement_meta
        sample["safety_risk"] = risk_data

        return sample

    def replace_items_in_file(self, input_file: str, output_file: str, max_workers: int = 24) -> Dict:
        """
        Process entire editing_plan.json file with parallel API calls.

        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file
            max_workers: Maximum number of parallel workers

        Returns:
            Statistics about replacements
        """
        print(f"Loading {input_file}...")

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)[1000:]

        total_files = len(data)
        print(f"Processing {total_files} samples with {max_workers} workers...")

        stats = {
            "total_samples": total_files,
            "replacement_attempts": 0,
            "replacements_success": 0,
            "replacements_skipped": 0,
            "item_usage": {}
        }

        failed_indices = []

        # Process each sample in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.replace_item_in_sample, sample): (i, sample)
                for i, sample in enumerate(data)
                if sample.get("safety_risk") is not None
            }

            with tqdm(total=len(future_to_index), desc="Replacing items") as pbar:
                for future in as_completed(future_to_index):
                    idx, sample = future_to_index[future]

                    try:
                        stats["replacement_attempts"] += 1
                        augmented_sample = future.result()

                        if augmented_sample.get("_replacement_meta", {}).get("replaced"):
                            stats["replacements_success"] += 1
                            item_name = augmented_sample["_replacement_meta"]["replacement_object"]
                            stats["item_usage"][item_name] = stats["item_usage"].get(item_name, 0) + 1
                        else:
                            stats["replacements_skipped"] += 1

                        data[idx] = augmented_sample

                    except KeyboardInterrupt:
                        print("\nProcess interrupted by user. Saving current results...")
                        raise
                    except Exception as e:
                        failed_indices.append({"index": idx, "error": str(e)})
                        stats["replacements_skipped"] += 1
                        traceback.print_exc()
                    finally:
                        pbar.update(1)

        # Save augmented data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nSaved augmented data to {output_file}")
        print(f"Stats: {stats}")
        if failed_indices:
            print(f"Failed cases: {failed_indices}")

        return stats


# ========================================================================
# Main Function
# ========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Item Replacement for Risk Grounding Data Augmentation"
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['generate_objects', 'replace'],
        help='Mode: generate_objects or replace'
    )
    parser.add_argument(
        '--augmentation_model',
        type=str,
        default='gemini-2.5-pro'
    )
    parser.add_argument(
        '--replace_model',
        type=str,
        default='Qwen/Qwen3-VL-235B-A22B-Thinking'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=24,
        help='Maximum number of parallel workers for API calls'
    )
    parser.add_argument(
        '--root_folder',
        type=str,
        default="data",
    )

    args = parser.parse_args()

    object_lists_dir = os.path.join(args.root_folder, "augmentation_object")

    if args.mode == 'generate_objects':
        # Generate object lists for all principles
        generator = ObjectListGenerator(
            model=args.augmentation_model,
            output_dir=object_lists_dir
        )

        num_principles = len(ACTION_TRIGGERED_PRINCIPLES)

        print(f"\nGenerating object lists for all {num_principles} action_triggered principles...")
        print("="*60)

        results = generator.generate_all_principles()

        print("\n" + "="*60)
        print(f"Object List Generation Complete!")
        print(f"  Principles processed: action_triggered")
        print(f"  Successfully generated: {len(results)}/{num_principles}")
        print("="*60)

    elif args.mode == 'replace':
        input_file = os.path.join(args.root_folder, "editing_plan.json")

        # Auto-generate output filename
        input_path = Path(input_file)
        output_path = str(input_path.parent / f"aug_{input_path.name}")

        # Replace objects in editing plans
        replacer = ItemReplacer(
            object_lists_dir=object_lists_dir,
            model=args.replace_model
        )

        stats = replacer.replace_items_in_file(input_file, output_path, args.max_workers)

        print("\n" + "="*60)
        print("Replacement Summary:")
        print(f"  Max workers: {args.max_workers}")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Replacements attempted: {stats['replacement_attempts']}")
        print(f"  Replacements successful: {stats['replacements_success']}")
        print(f"  Replacements skipped: {stats['replacements_skipped']}")
        print(f"  Unique items used: {len(stats['item_usage'])}")
        print(f"  Output file: {output_path}")
        print("="*60)


if __name__ == "__main__":
    main()
