"""
Evaluation utilities for model assessment.
"""

from typing import TypedDict, Literal

import roma
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from utils.loss import compute_angular_error


class EvaluationResults(TypedDict):
    """Results from model evaluation."""
    translation_error: float
    angular_error: float
    num_samples: int


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    rotation_representation: Literal["matrix", "quaternion"] = "matrix",
) -> EvaluationResults:
    """
    Evaluate model on a dataset.

    Args:
        model: The trained model (must accept two graph batches and return
               (pred_translation, pred_rotation) tuple)
        data_loader: DataLoader for the evaluation dataset
        rotation_representation: Either "matrix" (9D/3x3) or "quaternion" (4D).
            Determines how to process the model's rotation output.

    Returns:
        EvaluationResults dict with:
            - translation_error: Average L2 error per sample (translation)
            - angular_error: Angular error in degrees
            - num_samples: Number of samples evaluated
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    total_translation_error = 0.0
    total_angular_error = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            graph_a_batch = batch[0].to(device)
            graph_b_batch = batch[1].to(device)
            labels_translations = batch[2].to(device)
            labels_quat_xyzw = batch[3].to(device)

            pred_translation, pred_rotation_raw = model(graph_a_batch, graph_b_batch)

            # Process rotation based on representation
            if rotation_representation == "matrix":
                # Normalize the predicted matrix to a valid SO(3) rotation matrix
                pred_rotation = roma.special_procrustes(pred_rotation_raw)
            else:
                # Quaternion: normalize and convert to rotation matrix for angular error
                pred_quat_normalized = pred_rotation_raw / torch.norm(pred_rotation_raw, dim=-1, keepdim=True)
                pred_rotation = roma.unitquat_to_rotmat(pred_quat_normalized)

            # Convert ground truth quaternions to rotation matrices
            true_rotmat = roma.unitquat_to_rotmat(labels_quat_xyzw)

            batch_size = labels_translations.shape[0]

            # Translation L2 error per sample
            translation_errors = torch.norm(pred_translation - labels_translations, p=2, dim=-1)
            total_translation_error += translation_errors.sum().item()

            # Angular error in degrees
            angular_errors = compute_angular_error(pred_rotation, true_rotmat)
            total_angular_error += angular_errors.sum().item()

            num_samples += batch_size

    return EvaluationResults(
        translation_error=total_translation_error / num_samples,
        angular_error=total_angular_error / num_samples,
        num_samples=num_samples,
    )

