import argparse
import base64
import json
import os
from openai import OpenAI

PLANNING_WITH_SAFETY_PROMPT = """
You are a helpful robotic task planner. You should provide safe and effective task plans to guide the robotic arm in achieving the user's goals.

You will be given the following information:
- The input image provides an observation of the current environment. Use it to understand the spatial layout and identify potential objects, obstacles, or safety hazards relevant to planning.
- task_instruction: an everyday task that the indoor robot agent is required to complete
- reference_bounding_box
- safety_tips: if provided, the plan must follow the safety tips to ensure safe execution.

Available skills:
1. Move to [position] (x, y)
2. Pick up [object]
3. Grasp [object]
4. Place to [position] (x, y)
5. Pull [object]
6. Push [object]
7. Pour into [object]
8. Wipe [object]
9. Press [object]
10. Done

Rules:
- Assume the initial 2D coordinates of the robotic arm are (500, 500), normalized to 0-1000.
- You must use the move action to approach an object before operating on it.
- The final step must be Done.
- Output strictly valid JSON.

Your input:
- task_instruction: {task_instruction}
- reference_bounding_box: {bounding_box}
- safety_tips: {safety_tips}
""".strip()

PLANNING_PROMPT = """
You are a helpful robotic task planner. You should provide safe and effective task plans to guide the robotic arm in achieving the user's goals.

You will be given:
- the scene image
- task_instruction

Available skills:
1. Move to [position] (x, y)
2. Pick up [object]
3. Grasp [object]
4. Place to [position] (x, y)
5. Pull [object]
6. Push [object]
7. Pour into [object]
8. Wipe [object]
9. Press [object]
10. Done

Rules:
- Assume the initial 2D coordinates of the robotic arm are (500, 500), normalized to 0-1000.
- You must use the move action to approach an object before operating on it.
- The final step must be Done.
- Output strictly valid JSON.

Your input:
- task_instruction: {task_instruction}
""".strip()


def load_image_as_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def build_prompt(task_instruction: str, bounding_box: dict | None, safety_tips: str | None) -> str:
    if bounding_box is None:
        return PLANNING_PROMPT.format(task_instruction=task_instruction)
    return PLANNING_WITH_SAFETY_PROMPT.format(
        task_instruction=task_instruction,
        bounding_box=json.dumps(bounding_box, ensure_ascii=False, indent=2),
        safety_tips=safety_tips or "None",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate safe task plans conditioned on HomeGuard outputs.")
    parser.add_argument("--image", required=True, help="Path to the scene image")
    parser.add_argument("--task", required=True, help="Task instruction for the planner")
    parser.add_argument("--bbox-json", default=None, help="Optional JSON file with HomeGuard target/constraint bounding boxes")
    parser.add_argument("--safety-tips", default=None, help="Optional textual safety tips")
    parser.add_argument("--api-url", default=os.getenv("PLANNER_API_URL"), help="OpenAI-compatible planner endpoint")
    parser.add_argument("--api-key", default=os.getenv("PLANNER_API_KEY", "EMPTY"), help="Planner API key")
    parser.add_argument("--model", default=os.getenv("PLANNER_MODEL", "Qwen/Qwen3-VL-8B-Thinking"), help="Planner model name")
    args = parser.parse_args()

    if not args.api_url:
        raise ValueError("Planner endpoint missing. Set --api-url or PLANNER_API_URL.")

    bounding_box = None
    if args.bbox_json:
        with open(args.bbox_json, "r", encoding="utf-8") as f:
            bounding_box = json.load(f)

    prompt_text = build_prompt(args.task, bounding_box, args.safety_tips)
    client = OpenAI(api_key=args.api_key, base_url=args.api_url)
    base64_image = load_image_as_base64(args.image)
    response = client.chat.completions.create(
        model=args.model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
