import torch


def compute_angular_error(pred_rotmat, true_rotmat):
    """
    Compute angular error in degrees between predicted and true rotation matrices.

    Args:
        pred_rotmat: Predicted rotation matrices of shape (batch_size, 3, 3)
        true_rotmat: Ground truth rotation matrices of shape (batch_size, 3, 3)

    Returns:
        Angular errors in degrees, shape (batch_size,)
    """
    # Compute relative rotation: R_error = R_pred^T @ R_true
    R_error = torch.bmm(pred_rotmat.transpose(1, 2), true_rotmat)

    # Extract trace of rotation matrix
    trace = R_error[:, 0, 0] + R_error[:, 1, 1] + R_error[:, 2, 2]

    # Compute angle: theta = arccos((trace - 1) / 2)
    # Clamp to avoid numerical issues with arccos
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    # Angular error in radians, then convert to degrees
    angular_error_rad = torch.acos(cos_angle)
    angular_error_deg = torch.rad2deg(angular_error_rad)

    return angular_error_deg

