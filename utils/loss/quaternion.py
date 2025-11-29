import torch

def quaternion_loss(quat_pred, quat_gt):
    # 1. Compute the geodesic loss
    # Normalize the predicted quat, since the predictor may not have given us a clean unit vector!
    quat_pred_norm = torch.norm(quat_pred, dim=-1, keepdim=True)
    quat_pred_normalized = quat_pred / quat_pred_norm
    # Compute dot product between each gt and pred pair
    dot_prod = torch.sum(quat_gt * quat_pred_normalized, dim=-1)
    # Compute: 1 - (gt * pred)^2
    # geodesic_losses = 1 - dot_prod ** 2
    geodesic_losses = 1 - torch.abs(dot_prod)

    # 2. Compute the normalization loss (pred quaternion norm should strive to be 1!)
    norm_losses = (1 - quat_pred_norm) ** 2

    # 3. Add the loss components
    losses = geodesic_losses + norm_losses

    # 4. Compute the avg loss over the batch
    batch_loss = torch.mean(losses)
    return batch_loss