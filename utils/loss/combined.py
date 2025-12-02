import torch
import roma


def compute_combined_loss(pred_translation, pred_rotation, labels_translations, labels_quat_xyzw,
                          criterion_translation, criterion_rotation, lambda_translation):
    """
    Calculate translation, rotation, and combined loss.

    Args:
        pred_translation: Predicted translation vectors
        pred_rotation: Predicted rotation matrices
        labels_translations: Ground truth translation vectors
        labels_quat_xyzw: Ground truth quaternions (xyzw format)
        criterion_translation: Loss function for translation
        criterion_rotation: Loss function for rotation
        lambda_translation: Weight for translation loss in combined loss

    Returns:
        tuple: (combined_loss, translation_loss, rotation_loss)
    """
    # Calculate translation loss
    loss_trans = criterion_translation(pred_translation, labels_translations)

    # Calculate rotation loss - convert quaternions to rotation matrices
    labels_rotmat = roma.unitquat_to_rotmat(labels_quat_xyzw)
    loss_rotation = criterion_rotation(pred_rotation, labels_rotmat)

    # Combine loss
    loss = lambda_translation * loss_trans + loss_rotation

    return loss, loss_trans, loss_rotation


