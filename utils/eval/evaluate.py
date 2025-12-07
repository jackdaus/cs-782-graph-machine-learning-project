"""
Evaluation utilities for model assessment.
"""

from typing import TypedDict

import torch
from torch import nn
from torch_geometric.loader import DataLoader


class EvaluationResults(TypedDict):
    """Results from model evaluation."""
    mse_loss: float
    l2_error: float
    num_samples: int


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
) -> EvaluationResults:
    """
    Evaluate model on a dataset.

    Args:
        model: The trained model (must accept two graph batches and return
               (pred_translation, pred_rotation) tuple)
        data_loader: DataLoader for the evaluation dataset

    Returns:
        EvaluationResults dict with:
            - mse_loss: Mean squared error loss per sample
            - l2_error: Average L2 error per sample
            - num_samples: Number of samples evaluated
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    total_mse_loss = 0.0
    total_l2_error = 0.0
    num_samples = 0

    criterion = torch.nn.MSELoss(reduction='sum')

    with torch.no_grad():
        for batch in data_loader:
            graph_a_batch = batch[0].to(device)
            graph_b_batch = batch[1].to(device)
            labels_translations = batch[2].to(device)

            pred_translation, _ = model(graph_a_batch, graph_b_batch)

            # MSE loss (summed, not averaged)
            batch_size = labels_translations.shape[0]
            total_mse_loss += criterion(pred_translation, labels_translations).item()

            # L2 error per sample
            l2_errors = torch.norm(pred_translation - labels_translations, p=2, dim=-1)
            total_l2_error += l2_errors.sum().item()

            num_samples += batch_size

    return EvaluationResults(
        mse_loss=total_mse_loss / num_samples,
        l2_error=total_l2_error / num_samples,
        num_samples=num_samples,
    )

