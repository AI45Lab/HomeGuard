"""
Visualization Module for Risk Grounding Evaluation.

This module contains functions for visualizing ground truth vs predictions
with colored bounding boxes.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

def visualize_comparison(item: Dict, save_folder: str):
    """
    Visualize GT (green) vs Prediction (red) on the same image.

    Args:
        item: Dict containing image_path, prediction, gt_data
        target_model_name: Name of target model (for bbox format detection)
        save_folder: Path to save visualization

    Returns:
        Path to saved image, or None if error
    """
    try:
        # Load image
        image_path = item["image_path"]
        img = Image.open(image_path)
        width, height = img.size

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(img)

        # Parse GT bboxes (action_triggered format)
        gt_target_bbox = item["evaluation_metrics"].get("gt_target_bbox") or []
        gt_constraint_bbox = item["evaluation_metrics"].get("gt_constraint_bbox") or []

        pred_target_bbox = item["evaluation_metrics"].get("pred_target_bbox") or []
        pred_constraint_bbox = item["evaluation_metrics"].get("pred_constraint_bbox") or []

        # Draw GT bboxes (green, solid line)
        for bbox_item in gt_target_bbox:
            x1, y1, x2, y2 = bbox_item["bounding_box"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor='green', facecolor='none', linestyle='-'
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"GT: {bbox_item['label']} (target)",
                   color='green', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        for bbox_item in gt_constraint_bbox:
            x1, y1, x2, y2 = bbox_item["bounding_box"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor='green', facecolor='none', linestyle='-'
            )
            ax.add_patch(rect)
            label = bbox_item['label'] if bbox_item['label'] else 'constraint'
            ax.text(x1, y1 - 5, f"GT: {label}",
                   color='green', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        # Draw Prediction bboxes (red, dashed line)
        for bbox_item in pred_target_bbox:
            x1, y1, x2, y2 = bbox_item["bounding_box"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            ax.text(x2, y1 - 5, f"Pred: target",
                   color='red', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        for bbox_item in pred_constraint_bbox:
            x1, y1, x2, y2 = bbox_item["bounding_box"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            ax.text(x2, y1 - 5, f"Pred: constraint",
                   color='red', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=3, label='GT (Ground Truth)'),
            Line2D([0], [0], color='red', lw=3, linestyle='--', label='Prediction')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

        ax.set_title(f"GT (Green) vs Prediction (Red)", fontsize=14, fontweight='bold')
        ax.axis('off')

        file_name = os.path.basename(image_path)    
        save_path = os.path.join(save_folder, f"vis_{file_name}")

        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        return save_path

    except Exception as e:
        print(f"Error visualizing item {item.get('id', 'unknown')}: {e}")
        return None


def run_visualization_phase(eval_items: List[Dict],
                             target_model_name: str, save_folder: str,
                             max_workers: int = 8) -> int:
    """
    Run visualization phase in parallel.

    Args:
        eval_items: List of items containing predictions and ground truth
        target_model_name: Name of target model (for bbox format detection)
        save_folder: Path to save visualizations
        max_workers: Number of parallel workers

    Returns:
        Number of successfully visualized samples
    """
    vis_folder = os.path.join(save_folder, "visualizations")
    os.makedirs(vis_folder, exist_ok=True)

    for item in tqdm(eval_items):
        visualize_comparison(item, vis_folder)