"""
Generate synthetic SfM training data.

This script generates training data by:
1. Computing image features once and storing them in a shared cache
2. Storing samples as lightweight references to image IDs
3. Saving features and samples to separate files

Usage:
    # Use default config
    uv run python generate_data.py

    # Override parameters via CLI
    uv run python generate_data.py generation.num_samples=500 features.img_size=64

    # Use different paths
    uv run python generate_data.py paths.model=./other/model paths.output=data/custom

The output files are:
    - {output_dir}/image_features.pt: Shared image feature cache
    - {output_dir}/samples.pt: Lightweight sample data
"""

import logging
import pathlib
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import hydra
import numpy as np
import pycolmap
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import utils.subset
from utils.loaders.efficient import (
    EfficientSfmPairDataset,
    ImageFeatureCache,
    SfmSample,
    compute_edge_index,
)

log = logging.getLogger(__name__)


# ============================================================================
# Structured Configuration Classes
# ============================================================================


@dataclass
class PathsConfig:
    """Configuration for input/output paths."""
    model: str = MISSING
    images: str = MISSING
    output: str = "data/generated"


@dataclass
class GenerationConfig:
    """Configuration for data generation parameters."""
    num_samples: int = 1000
    subset_size: int = 15
    random_seed: int = 42


@dataclass
class FeaturesConfig:
    """Configuration for image feature extraction."""
    include_image_features: bool = True
    img_size: int = 100


@dataclass
class TransformConfig:
    """Configuration for pose transformation/augmentation."""
    translation_min: float = -10.0
    translation_max: float = 10.0
    enable_rotation_perturbation: bool = True


@dataclass
class DataGenerationConfig:
    """Root configuration for data generation."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)


# Register config schema with Hydra
cs = ConfigStore.instance()
cs.store(name="data_generation_schema", node=DataGenerationConfig)


# ============================================================================
# Data Generation Functions
# ============================================================================


def generate_sfm_data(cfg: DictConfig) -> EfficientSfmPairDataset:
    """
    Generate SfM training data from a COLMAP reconstruction.

    Args:
        cfg: Hydra configuration object containing all parameters

    Returns:
        EfficientSfmPairDataset with generated samples
    """
    # Convert paths
    model_path = pathlib.Path(cfg.paths.model)
    image_dir = pathlib.Path(cfg.paths.images)
    output_dir = pathlib.Path(cfg.paths.output)

    # Set random seeds
    random.seed(cfg.generation.random_seed)
    np.random.seed(cfg.generation.random_seed)

    # Load the Colmap data
    reconstruction = pycolmap.Reconstruction(model_path)
    log.info(f"Loaded reconstruction with {len(reconstruction.images)} images")

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Precompute and save image features (only if needed)
    features_path = output_dir / 'image_features.pt'
    if cfg.features.include_image_features:
        log.info(f"Precomputing image features at {cfg.features.img_size}x{cfg.features.img_size}...")
        cache = ImageFeatureCache(reconstruction, image_dir, cfg.features.img_size)
        cache.precompute_all()
        cache.save(features_path)
        image_features = cache.get_all_features()
    else:
        log.info("Skipping image features (include_image_features=false)")
        image_features = {}

    # Step 2: Generate samples (lightweight, only IDs and transforms)
    log.info(f"Generating {cfg.generation.num_samples} samples...")
    translation_range = (cfg.transform.translation_min, cfg.transform.translation_max)
    samples = _generate_samples(
        reconstruction=reconstruction,
        num_samples=cfg.generation.num_samples,
        subset_size=cfg.generation.subset_size,
        translation_range=translation_range,
        enable_rotation_perturbation=cfg.transform.enable_rotation_perturbation,
    )

    # Create dataset
    dataset = EfficientSfmPairDataset(samples, image_features, cfg.features.include_image_features)

    # Save with metadata
    metadata = {
        'num_samples': cfg.generation.num_samples,
        'subset_size': cfg.generation.subset_size,
        'include_image_features': cfg.features.include_image_features,
        'translation_range': translation_range,
        'enable_rotation_perturbation': cfg.transform.enable_rotation_perturbation,
        'random_seed': cfg.generation.random_seed,
        'model_path': str(model_path),
        'image_dir': str(image_dir),
    }
    if cfg.features.include_image_features:
        metadata['img_size'] = cfg.features.img_size

    samples_path = output_dir / 'samples.pt'
    dataset.save(samples_path, metadata=metadata)

    _log_summary(cfg, output_dir, features_path, samples_path)

    return dataset


def _generate_samples(
    reconstruction: pycolmap.Reconstruction,
    num_samples: int,
    subset_size: int,
    translation_range: Tuple[float, float],
    enable_rotation_perturbation: bool,
) -> List[SfmSample]:
    """
    Generate lightweight SfmSample objects.

    Args:
        reconstruction: COLMAP reconstruction
        num_samples: Number of samples to generate
        subset_size: Size of each image subset
        translation_range: Range for random translations (min, max)
        enable_rotation_perturbation: Whether to apply random rotation

    Returns:
        List of SfmSample objects
    """
    samples = []

    pbar = tqdm(range(num_samples), desc="Generating samples")
    for _ in pbar:
        # Create random subsets using distance-based heuristic
        a_ids, b_ids = utils.subset.create_image_subsets_from_distance(
            reconstruction, subset_size, random_captains=True
        )

        # Compute edge indices
        edge_index_a = compute_edge_index(reconstruction, a_ids)
        edge_index_b = compute_edge_index(reconstruction, b_ids)

        # Get poses as (quat, translation) tuples
        poses_a = _get_poses(reconstruction, a_ids)
        poses_b = _get_poses(reconstruction, b_ids)

        # Generate random transformation
        random_translation = torch.empty(3).uniform_(*translation_range)
        if enable_rotation_perturbation:
            random_rotation = Rotation.random().as_quat()  # xyzw format
        else:
            random_rotation = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion (xyzw)
        random_transform = pycolmap.Rigid3d(random_rotation, random_translation.numpy())

        # Apply transformation to poses_b
        transformed_poses_b = _apply_transform(poses_b, random_transform)

        samples.append(SfmSample(
            image_ids_a=a_ids,
            image_ids_b=b_ids,
            edge_index_a=edge_index_a,
            edge_index_b=edge_index_b,
            poses_a=poses_a,
            poses_b=transformed_poses_b,
            label_translation=random_translation,
            label_quat_xyzw=torch.from_numpy(random_rotation).float(),
        ))

    return samples


def _get_poses(
    reconstruction: pycolmap.Reconstruction,
    image_ids: List[int],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract poses as (quaternion, translation) tuples from reconstruction."""
    poses = []
    for image_id in image_ids:
        world_from_cam = reconstruction.images[image_id].cam_from_world().inverse()
        quat = np.array(world_from_cam.rotation.quat)
        trans = np.array(world_from_cam.translation)
        poses.append((quat, trans))
    return poses


def _apply_transform(
    poses: List[Tuple[np.ndarray, np.ndarray]],
    transform: pycolmap.Rigid3d,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Apply a rigid transformation to a list of poses."""
    transformed_poses = []
    for quat, trans in poses:
        original_rigid = pycolmap.Rigid3d(quat, trans)
        transformed_rigid = transform * original_rigid
        new_quat = np.array(transformed_rigid.rotation.quat)
        new_trans = np.array(transformed_rigid.translation)
        transformed_poses.append((new_quat, new_trans))
    return transformed_poses


def _log_summary(
    cfg: DictConfig,
    output_dir: pathlib.Path,
    features_path: pathlib.Path,
    samples_path: pathlib.Path,
) -> None:
    """Log a summary of the generated data."""
    log.info("=" * 50)
    log.info("Data generation complete!")
    log.info("=" * 50)
    log.info(f"Output directory: {output_dir}")
    if cfg.features.include_image_features:
        log.info(f"Image features: {features_path}")
    log.info(f"Samples: {samples_path}")
    log.info("=" * 50)
    log.info(f"Generated {cfg.generation.num_samples} samples")
    log.info(f"Subset size: {cfg.generation.subset_size}")
    log.info(f"Include image features: {cfg.features.include_image_features}")
    if cfg.features.include_image_features:
        log.info(f"Image size: {cfg.features.img_size}x{cfg.features.img_size}")
    log.info(f"Rotation perturbation: {cfg.transform.enable_rotation_perturbation}")


def load_dataset(
    output_dir: pathlib.Path,
    include_image_features: bool,
) -> EfficientSfmPairDataset:
    """
    Load a previously generated dataset.

    Args:
        output_dir: Directory containing the generated data
        include_image_features: Whether to include image features in node features

    Returns:
        Loaded EfficientSfmPairDataset
    """
    samples_path = output_dir / 'samples.pt'
    features_path = output_dir / 'image_features.pt'

    return EfficientSfmPairDataset.load(samples_path, features_path, include_image_features)


# ============================================================================
# Hydra Entry Point
# ============================================================================


@hydra.main(version_base=None, config_path="conf", config_name="data_generation_example")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for data generation with Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    # Log the configuration being used
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Convert paths to pathlib.Path objects for validation
    model_path = pathlib.Path(cfg.paths.model)
    image_dir = pathlib.Path(cfg.paths.images)

    # Validate paths exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    # Generate data
    generate_sfm_data(cfg)


if __name__ == '__main__':
    main()
