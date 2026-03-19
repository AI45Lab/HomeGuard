import argparse
import os
from evaluation.utils import add_sys_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay or visualize low-level trajectories with RoboBrain.")
    parser.add_argument("--prompt", required=True, help="Trajectory instruction for RoboBrain")
    parser.add_argument("--image", required=True, help="Scene image path")
    parser.add_argument("--model-path", default=os.getenv("ROBOBRAIN_MODEL_PATH"), help="Local RoboBrain checkpoint")
    parser.add_argument("--third-party-root", default=os.path.join(os.path.dirname(__file__), "..", "third_party", "Robobrain2.5"), help="Path to the RoboBrain2.5 repository")
    parser.add_argument("--plot", action="store_true", help="Whether to render the predicted trajectory")
    args = parser.parse_args()

    if not args.model_path:
        raise ValueError("RoboBrain checkpoint missing. Set --model-path or ROBOBRAIN_MODEL_PATH.")

    with add_sys_path(args.third_party_root):
        from inference import UnifiedInference

        model = UnifiedInference(args.model_path)
        prediction = model.inference(args.prompt, args.image, task="trajectory", plot=args.plot, do_sample=False)
        print(prediction)


if __name__ == "__main__":
    main()
