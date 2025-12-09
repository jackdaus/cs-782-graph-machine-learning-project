"""
Loss function factory for rotation learning experiments.

This module provides a unified interface for selecting and instantiating
rotation loss functions based on configuration.
"""

import torch
from torch import nn
import roma

from .quaternion import quaternion_dot_loss


def frobenius_loss(pred_rotmat: torch.Tensor, true_rotmat: torch.Tensor) -> torch.Tensor:
    """
    Compute Frobenius norm loss between predicted and true rotation matrices.

    Args:
        pred_rotmat: Predicted rotation matrices of shape (batch_size, 3, 3)
        true_rotmat: Ground truth rotation matrices of shape (batch_size, 3, 3)

    Returns:
        Mean Frobenius norm loss over the batch
    """
    diff = pred_rotmat - true_rotmat
    # Frobenius norm per sample, then mean over batch
    frob_norms = torch.norm(diff, p='fro', dim=(-2, -1))
    return frob_norms.mean()



class RotationLossWrapper(nn.Module):
    """
    Wrapper that handles both quaternion and rotation matrix inputs,
    converting as needed before computing the underlying loss.
    """
    def __init__(self, loss_fn, rotation_representation: str):
        """
        Args:
            loss_fn: The underlying loss function
            rotation_representation: Either "quaternion" or "matrix"
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.rotation_representation = rotation_representation

    def forward(self, pred_rotation: torch.Tensor, labels_quat_xyzw: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation loss.

        Args:
            pred_rotation: Predicted rotation (quaternion or matrix depending on representation)
            labels_quat_xyzw: Ground truth quaternion in xyzw format

        Returns:
            Scalar loss value
        """
        if self.rotation_representation == "quaternion":
            # pred_rotation is quaternion, labels are quaternion
            return self.loss_fn(pred_rotation, labels_quat_xyzw)
        else:
            # pred_rotation is rotation matrix, convert labels to matrix
            labels_rotmat = roma.unitquat_to_rotmat(labels_quat_xyzw)
            return self.loss_fn(pred_rotation, labels_rotmat)


# Registry of available rotation loss functions
ROTATION_LOSS_REGISTRY = {
    # For rotation matrix representation
    "l1": lambda: nn.L1Loss(),
    "mse": lambda: nn.MSELoss(),
    "frobenius": lambda: frobenius_loss,
    # For quaternion representation
    "quaternion_dot": lambda: quaternion_dot_loss,
}


def get_rotation_loss(loss_name: str, rotation_representation: str) -> RotationLossWrapper:
    """
    Get a rotation loss function by name.

    Args:
        loss_name: Name of the loss function. Options:
            - "l1": L1 loss on quaternions or rotation matrices
            - "mse": MSE loss on quaternions or rotation matrices
            - "frobenius": Frobenius norm loss on rotation matrices
            - "quaternion_dot": loss on quaternions using dot product
        rotation_representation: Either "quaternion" or "matrix"

    Returns:
        RotationLossWrapper that handles the specified rotation representation

    Raises:
        ValueError: If loss_name is not found in registry or incompatible with representation
    """
    if loss_name not in ROTATION_LOSS_REGISTRY:
        available = ", ".join(ROTATION_LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown rotation loss '{loss_name}'. Available: {available}")

    # Validate compatibility between loss function and rotation representation
    matrix_losses = {"frobenius"}
    quaternion_losses = {"quaternion_dot"}

    if rotation_representation == "matrix" and loss_name in quaternion_losses:
        raise ValueError(
            f"Loss '{loss_name}' is only compatible with rotation_representation='quaternion', "
            f"but got rotation_representation='matrix'. "
            f"Use one of {matrix_losses} for matrix representation."
        )
    if rotation_representation == "quaternion" and loss_name in matrix_losses:
        raise ValueError(
            f"Loss '{loss_name}' is only compatible with rotation_representation='matrix', "
            f"but got rotation_representation='quaternion'. "
            f"Use 'quaternion' loss for quaternion representation."
        )

    loss_fn = ROTATION_LOSS_REGISTRY[loss_name]()
    return RotationLossWrapper(loss_fn, rotation_representation)

