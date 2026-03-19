"""
Weighted reward wrapper for Risk Grounding RFT training.
Allows applying different weights to each reward function.
Uses batch rewards to avoid repeated parsing.
"""

from typing import Callable, List, Dict
from rewards import format_reward, RiskGroundingRewards


class WeightedRewards:
    """
    Wrapper for reward functions with configurable weights.
    """

    # Default reward weights
    DEFAULT_WEIGHTS = {
        "safe_accuracy": 1.0,
        "safety_hazard_match": 1.0,
        "principle_accuracy": 1.0,
        "iou_target_object": 1.0,
        "iou_constraint_object": 1.0,
        "format": 0.5,
    }

    def __init__(self, embedding_model_path: str = "checkpoints/all-MiniLM-L6-v2",
                 weights: Dict[str, float] = None):
        """
        Initialize weighted reward calculator.

        Args:
            embedding_model_path: Path to sentence embedding model
            weights: Dictionary mapping reward names to weights.
                    If None, uses DEFAULT_WEIGHTS.
        """
        self.calculator = RiskGroundingRewards()
        self.weights = weights or self.DEFAULT_WEIGHTS

    def safe_accuracy_reward(self, completions, solution, **kwargs):
        """Safe accuracy reward with weight applied."""
        base_rewards = self.calculator.safe_accuracy_reward(completions, solution, **kwargs)
        weight = self.weights.get("safe_accuracy", 1.0)
        return [r * weight for r in base_rewards]

    def safety_hazard_match_reward(self, completions, solution, **kwargs):
        """Safety hazard match reward with weight applied."""
        base_rewards = self.calculator.safety_hazard_match_reward(completions, solution, **kwargs)
        weight = self.weights.get("safety_hazard_match", 1.0)
        return [r * weight for r in base_rewards]

    def principle_accuracy_reward(self, completions, solution, **kwargs):
        """Principle accuracy reward with weight applied."""
        base_rewards = self.calculator.principle_accuracy_reward(completions, solution, **kwargs)
        weight = self.weights.get("principle_accuracy", 1.0)
        return [r * weight for r in base_rewards]

    def iou_target_object_reward(self, completions, solution, **kwargs):
        """Target object IoU reward with weight applied."""
        base_rewards = self.calculator.iou_target_object_reward(completions, solution, **kwargs)
        weight = self.weights.get("iou_target_object", 1.0)
        return [r * weight for r in base_rewards]

    def iou_constraint_object_reward(self, completions, solution, **kwargs):
        """Constraint object IoU reward with weight applied."""
        base_rewards = self.calculator.iou_constraint_object_reward(completions, solution, **kwargs)
        weight = self.weights.get("iou_constraint_object", 1.0)
        return [r * weight for r in base_rewards]

    def format_reward(self, completions, **kwargs):
        """Format reward with weight applied."""
        base_rewards = format_reward(completions, **kwargs)
        weight = self.weights.get("format", 1.0)
        return [r * weight for r in base_rewards]


def get_weighted_reward_registry(embedding_model_path: str, weights: Dict[str, float] = None):
    """
    Get a dictionary of weighted reward functions.

    Args:
        embedding_model_path: Path to sentence embedding model
        weights: Dictionary mapping reward names to weights

    Returns:
        Dictionary of weighted reward functions
    """
    weighted = WeightedRewards(embedding_model_path, weights)
    return {
        "safe_accuracy": weighted.safe_accuracy_reward,
        "safety_hazard_match": weighted.safety_hazard_match_reward,
        "principle_accuracy": weighted.principle_accuracy_reward,
        "iou_target_object": weighted.iou_target_object_reward,
        "iou_constraint_object": weighted.iou_constraint_object_reward,
        "format": weighted.format_reward,
    }


# ========================================================================
# Preset weight configurations
# ========================================================================

# Focus on safety classification (safe/unsafe)
SAFETY_FOCUSED_WEIGHTS = {
    "safe_accuracy": 2.0,
    "safety_hazard_match": 0.5,
    "principle_accuracy": 0.5,
    "iou_target_object": 1.0,
    "iou_constraint_object": 1.0,
    "format": 0.5,
}

# Focus on localization accuracy (bounding box)
LOCALIZATION_FOCUSED_WEIGHTS = {
    "safe_accuracy": 1.0,
    "safety_hazard_match": 0.5,
    "principle_accuracy": 0.5,
    "iou_target_object": 2.0,
    "iou_constraint_object": 2.0,
    "format": 0.5,
}

# Focus on safety hazard description quality
DESCRIPTION_FOCUSED_WEIGHTS = {
    "safe_accuracy": 1.0,
    "safety_hazard_match": 2.0,
    "principle_accuracy": 1.0,
    "iou_target_object": 1.0,
    "iou_constraint_object": 1.0,
    "format": 0.5,
}

# Focus on principle classification accuracy
PRINCIPLE_FOCUSED_WEIGHTS = {
    "safe_accuracy": 1.0,
    "safety_hazard_match": 1.0,
    "principle_accuracy": 2.0,
    "iou_target_object": 1.0,
    "iou_constraint_object": 1.0,
    "format": 0.5,
}

# Balanced weights (equal importance)
BALANCED_WEIGHTS = {
    "safe_accuracy": 1.0,
    "safety_hazard_match": 1.0,
    "principle_accuracy": 1.0,
    "iou_target_object": 1.0,
    "iou_constraint_object": 1.0,
    "format": 0.5,  # Format is less important
}

# Focus on target object localization (action_triggered)
TARGET_OBJECT_FOCUSED_WEIGHTS = {
    "safe_accuracy": 1.0,
    "safety_hazard_match": 0.5,
    "principle_accuracy": 0.5,
    "iou_target_object": 2.0,
    "iou_constraint_object": 1.0,
    "format": 0.5,
}

# Focus on constraint object localization (action_triggered)
CONSTRAINT_OBJECT_FOCUSED_WEIGHTS = {
    "safe_accuracy": 1.0,
    "safety_hazard_match": 0.5,
    "principle_accuracy": 0.5,
    "iou_target_object": 1.0,
    "iou_constraint_object": 2.0,
    "format": 0.5,
}
