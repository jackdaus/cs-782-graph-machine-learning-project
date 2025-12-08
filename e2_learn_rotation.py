"""
e2_learn_rotation.py

Experiment 2: Learn to predict rotation between two graphs.

Usage:
    uv run e2_learn_rotation.py --config-name e2_0_base.yaml
"""

import pathlib
import random
import sys
from datetime import datetime
from pprint import pprint

import hydra
import numpy as np
import roma
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from generate_data import get_dataset
from models.models import get_model
from utils.eval import evaluate
from utils.loss import compute_combined_loss, compute_angular_error


def log(msg: str = ""):
    """Print with immediate flush for real-time output."""
    print(msg, flush=True)


@hydra.main(version_base=None, config_path="conf/e2", config_name="e2_0_base")
def main(cfg: DictConfig):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(43)
    random.seed(43)

    # Load dataset
    dataset = get_dataset(cfg.dataset.name, cfg.dataset.include_image_features)
    log(f"Loaded dataset with {len(dataset)} samples")
    log("Metadata:")
    pprint(dataset.metadata)
    sys.stdout.flush()

    # Split into train/val/test (70-20-10)
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - val_size  # Remainder for test

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Print sample info
    graph_a, graph_b, label_trans, label_quat = dataset[0]
    log(f"\nGraph A:")
    log(f"  - Node features shape: {graph_a.x.shape}")
    log(f"  - Edge index shape: {graph_a.edge_index.shape}")
    log(f"  - Number of nodes: {graph_a.x.shape[0]}")
    log(f"  - Number of edges: {graph_a.edge_index.shape[1]}")
    log(f"\nGraph B:")
    log(f"  - Node features shape: {graph_b.x.shape}")
    log(f"  - Edge index shape: {graph_b.edge_index.shape}")
    log(f"\nLabels:")
    log(f"  - Translation: {label_trans}")
    log(f"  - Quaternion (xyzw): {label_quat}")

    # Log configuration
    log("\nConfig:")
    log(OmegaConf.to_yaml(cfg))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, cfg.training.batch_size, shuffle=False)

    log(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    log(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")

    # Setup output directory
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_name = HydraConfig.get().job.config_name
    run_dir = pathlib.Path(f'runs/{config_name}/{timestamp_str}')
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup TensorBoard
    # View results by running: tensorboard --logdir=runs
    writer = SummaryWriter(str(run_dir))
    writer.add_text("config", OmegaConf.to_yaml(cfg))

    # Get feature dimension from first sample
    sample_graph_a, _, _, _ = dataset[0]
    num_node_features = sample_graph_a.x.shape[1]

    # Define the model
    model = get_model(
        cfg.model.name,
        num_node_features=num_node_features,
        graph_embedding_dim=cfg.model.embedding_size
    ).to(device)
    writer.add_text("model", str(model))

    # Optimizer and scheduler
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = None
    if cfg.training.scheduler.enabled:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.training.scheduler.step_size,
            gamma=cfg.training.scheduler.gamma
        )

    # Loss functions
    criterion_translation = torch.nn.MSELoss()
    criterion_rotation = torch.nn.L1Loss()

    # Training loop
    epoch_losses_train = []
    epoch_losses_val = np.full(cfg.training.num_epochs, np.nan)

    with tqdm(range(cfg.training.num_epochs), desc="Training progress") as pbar:
        for epoch in pbar:
            model.train()
            train_loss_all = 0.0
            train_loss_translation = 0.0
            train_loss_rotation = 0.0
            train_angular_error = 0.0
            train_frobenius_norm = 0.0
            train_translation_error = 0.0

            for batch in train_loader:
                # Move data to the GPU
                graph_a_batch = batch[0].to(device)
                graph_b_batch = batch[1].to(device)
                labels_translations = batch[2].to(device)
                labels_quat_xyzw = batch[3].to(device)

                optimizer.zero_grad()
                pred_translation, pred_rotation_raw = model(graph_a_batch, graph_b_batch)

                # "Normalize" the predicted matrix so that it is a valid SO(3) rotation matrix
                pred_rotation = roma.special_procrustes(pred_rotation_raw)

                # Calculate Frobenius norm between pred_rotation_raw and pred_rotation
                # This measures how far the raw prediction is from the SO(3) manifold
                frobenius_norm = torch.norm(pred_rotation_raw - pred_rotation, p='fro', dim=(-2, -1)).mean()

                # Calculate losses using the helper function
                loss, loss_trans, loss_rotation = compute_combined_loss(
                    pred_translation, pred_rotation, labels_translations, labels_quat_xyzw,
                    criterion_translation, criterion_rotation, cfg.training.lambda_translation
                )

                # Add Frobenius norm as regularization term
                lambda_frobenius = cfg.training.lambda_frobenius
                loss = loss + lambda_frobenius * frobenius_norm

                # Calculate angular error
                true_rotmat = roma.unitquat_to_rotmat(labels_quat_xyzw)
                angular_error = compute_angular_error(pred_rotation, true_rotmat)
                train_angular_error += angular_error.mean().item()

                # Calculate translation error (L2 norm)
                translation_error = torch.norm(pred_translation - labels_translations, p=2, dim=-1).mean()
                train_translation_error += translation_error.item()

                # Compute gradient and update weights
                loss.backward()
                optimizer.step()

                train_loss_all += loss.item()
                train_loss_translation += loss_trans.item()
                train_loss_rotation += loss_rotation.item()
                train_frobenius_norm += frobenius_norm.item()

            # Average losses
            avg_loss = train_loss_all / len(train_loader)
            avg_loss_translation = train_loss_translation / len(train_loader)
            avg_loss_rotation = train_loss_rotation / len(train_loader)
            avg_angular_error = train_angular_error / len(train_loader)
            avg_frobenius_norm = train_frobenius_norm / len(train_loader)
            avg_translation_error = train_translation_error / len(train_loader)

            # Log to TensorBoard
            writer.add_scalar('loss_all/train', avg_loss, epoch)
            writer.add_scalar('loss_translation/train', avg_loss_translation, epoch)
            writer.add_scalar('loss_rotation/train', avg_loss_rotation, epoch)
            writer.add_scalar('angular_error/train', avg_angular_error, epoch)
            writer.add_scalar('frobenius_norm/train', avg_frobenius_norm, epoch)
            writer.add_scalar('translation_error/train', avg_translation_error, epoch)

            epoch_losses_train.append(avg_loss)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'loss_trans': f'{avg_loss_translation:.4f}',
                              'loss_rot:': f'{avg_loss_rotation:.4f}', 'ang_err': f'{avg_angular_error:.2f}°',
                              'frob': f'{avg_frobenius_norm:.4f}' })

            # Learning rate decay
            if scheduler is not None:
                scheduler.step()

            # Validation
            if epoch % 10 == 0:
                model.eval()
                val_loss = 0.0
                val_loss_translation = 0.0
                val_loss_rotation = 0.0
                val_angular_error = 0.0
                val_frobenius_norm = 0.0
                val_translation_error = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        graph_a_batch = batch[0].to(device)
                        graph_b_batch = batch[1].to(device)
                        labels_translations = batch[2].to(device)
                        labels_quat_xyzw = batch[3].to(device)
                        pred_translation, pred_rotation_raw = model(graph_a_batch, graph_b_batch)
                        # "Normalize" the predicted matrix so that it is a valid SO(3) rotation matrix
                        pred_rotation = roma.special_procrustes(pred_rotation_raw)

                        # Calculate Frobenius norm between pred_rotation_raw and pred_rotation
                        frobenius_norm = torch.norm(pred_rotation_raw - pred_rotation, p='fro', dim=(-2, -1)).mean()

                        # Calculate losses using the helper function
                        loss, loss_trans, loss_rotation = compute_combined_loss(
                            pred_translation, pred_rotation, labels_translations, labels_quat_xyzw,
                            criterion_translation, criterion_rotation, cfg.training.lambda_translation
                        )

                        # Add Frobenius norm as regularization term
                        lambda_frobenius = cfg.training.lambda_frobenius  # Weight for the regularization term
                        loss = loss + lambda_frobenius * frobenius_norm

                        # Calculate angular error
                        true_rotmat = roma.unitquat_to_rotmat(labels_quat_xyzw)
                        angular_error = compute_angular_error(pred_rotation, true_rotmat)
                        val_angular_error += angular_error.mean().item()

                        # Calculate translation error (L2 norm)
                        translation_error = torch.norm(pred_translation - labels_translations, p=2, dim=-1).mean()
                        val_translation_error += translation_error.item()

                        val_loss += loss.item()
                        val_loss_translation += loss_trans.item()
                        val_loss_rotation += loss_rotation.item()
                        val_frobenius_norm += frobenius_norm.item()
                    avg_val_loss = val_loss / len(val_loader)
                    avg_val_loss_translation = val_loss_translation / len(val_loader)
                    avg_val_loss_rotation = val_loss_rotation / len(val_loader)
                    avg_val_angular_error = val_angular_error / len(val_loader)
                    avg_val_frobenius_norm = val_frobenius_norm / len(val_loader)
                    avg_val_translation_error = val_translation_error / len(val_loader)
                    epoch_losses_val[epoch] = avg_val_loss
                    writer.add_scalar('loss_all/val', avg_val_loss, epoch)
                    writer.add_scalar('loss_translation/val', avg_val_loss_translation, epoch)
                    writer.add_scalar('loss_rotation/val', avg_val_loss_rotation, epoch)
                    writer.add_scalar('angular_error/val', avg_val_angular_error, epoch)
                    writer.add_scalar('frobenius_norm/val', avg_val_frobenius_norm, epoch)
                    writer.add_scalar('translation_error/val', avg_val_translation_error, epoch)

    log(f"\nTraining complete. Final train loss: {epoch_losses_train[-1]:.4f}")
    log(f"TensorBoard logs saved to: {run_dir}")

    # Save model checkpoint
    checkpoint_path = run_dir / "model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': OmegaConf.to_container(cfg),
        'num_node_features': num_node_features,
        'final_train_loss': epoch_losses_train[-1],
    }, checkpoint_path)
    log(f"Model checkpoint saved to: {checkpoint_path}")

    # Evaluate on test set
    log("\n" + "=" * 50)
    log("Evaluating on held-out TEST set...")
    log("=" * 50)

    test_loader = DataLoader(test_dataset, cfg.training.batch_size, shuffle=False)
    test_results = evaluate(model, test_loader)

    log(f"\nTest Results:")
    log(f"  - Translation Error: {test_results['translation_error']:.6f}")
    log(f"  - Angular Error: {test_results['angular_error']:.2f}°")
    log(f"  - Num samples: {test_results['num_samples']}")

    # Log test metrics to TensorBoard
    writer.add_scalar('translation_error/test', test_results['translation_error'], cfg.training.num_epochs)
    writer.add_scalar('angular_error/test', test_results['angular_error'], cfg.training.num_epochs)
    writer.close()

    return test_results




if __name__ == "__main__":
    main()

