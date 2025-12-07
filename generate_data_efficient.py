"""
Generate synthetic SfM training data efficiently.

This script generates data in a memory-efficient manner by:
1. Computing image features once and storing them in a shared cache
2. Storing samples as lightweight references to image IDs
3. Saving features and samples to separate files

Usage:
    python generate_data_efficient.py

The output files are:
    - data/efficient/image_features.pt: Shared image feature cache
    - data/efficient/samples.pt: Lightweight sample data
"""

import pathlib
import random
from typing import Tuple

import numpy as np
import torch
import pycolmap
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import utils.subset
from utils.loaders.efficient import (
    ImageFeatureCache,
    SfmSample,
    EfficientSfmPairDataset,
    compute_edge_index,
)


def generate_sfm_data_efficient(
    model_path: pathlib.Path,
    image_dir: pathlib.Path,
    num_samples: int = 10,
    subset_size: int = 15,
    include_image_features: bool = True,
    img_size: int = 100,
    translation_range: Tuple[float, float] = (-10.0, 10.0),
    random_seed: int = 42,
    output_dir: pathlib.Path = None,
) -> EfficientSfmPairDataset:
    """
    Generate data efficiently - features computed once, samples store references.

    Args:
        model_path: Path to COLMAP reconstruction model
        image_dir: Path to source images directory
        num_samples: Number of data samples to generate
        subset_size: Size of each image subset
        include_image_features: Whether to include image features in node features
        img_size: Resize dimension for images
        translation_range: Range for random translations (min, max)
        random_seed: Random seed for reproducibility
        output_dir: Directory to save the generated data

    Returns:
        EfficientSfmPairDataset with generated samples
    """
    # Set random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load the Colmap data
    reconstruction = pycolmap.Reconstruction(model_path)
    print(f"Loaded reconstruction with {len(reconstruction.images)} images")

    # Setup output directory
    output_dir = output_dir or pathlib.Path('data/efficient')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Precompute and save image features ONCE
    print(f"\nPrecomputing image features at {img_size}x{img_size}...")
    cache = ImageFeatureCache(reconstruction, image_dir, img_size)
    cache.precompute_all()

    features_path = output_dir / 'image_features.pt'
    cache.save(features_path)

    # Calculate feature size
    sample_feature = next(iter(cache._cache.values()))
    feature_size_mb = (len(cache._cache) * sample_feature.numel() * 4) / (1024 * 1024)
    print(f"Image features size: {feature_size_mb:.2f} MB (stored once)")

    # Step 2: Generate samples (lightweight, only IDs and transforms)
    print(f"\nGenerating {num_samples} samples...")
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
        def get_poses(image_ids):
            poses = []
            for image_id in image_ids:
                world_from_cam = reconstruction.images[image_id].cam_from_world().inverse()
                quat = np.array(world_from_cam.rotation.quat)
                trans = np.array(world_from_cam.translation)
                poses.append((quat, trans))
            return poses

        poses_a = get_poses(a_ids)
        poses_b = get_poses(b_ids)

        # Generate random transformation
        random_translation = torch.empty(3).uniform_(*translation_range)
        random_rotation = Rotation.random().as_quat()  # xyzw format
        random_transform = pycolmap.Rigid3d(random_rotation, random_translation.numpy())

        # Apply transformation to poses_b
        transformed_poses_b = []
        for quat, trans in poses_b:
            original_rigid = pycolmap.Rigid3d(quat, trans)
            transformed_rigid = random_transform * original_rigid
            new_quat = np.array(transformed_rigid.rotation.quat)
            new_trans = np.array(transformed_rigid.translation)
            transformed_poses_b.append((new_quat, new_trans))

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

    # Create dataset with metadata
    dataset = EfficientSfmPairDataset(samples, cache._cache, include_image_features)

    # Save with metadata
    metadata = {
        'num_samples': num_samples,
        'subset_size': subset_size,
        'include_image_features': include_image_features,
        'img_size': img_size,
        'translation_range': translation_range,
        'random_seed': random_seed,
        'model_path': str(model_path),
        'image_dir': str(image_dir),
    }

    samples_path = output_dir / 'samples.pt'
    dataset.save(samples_path, metadata=metadata)

    # Calculate samples size (rough estimate)
    # Each sample stores: 2 lists of ~15 ints, 2 edge tensors, 2 pose lists, 2 label tensors
    samples_size_mb = samples_path.stat().st_size / (1024 * 1024)

    print(f"\n{'='*50}")
    print(f"Data generation complete!")
    print(f"{'='*50}")
    print(f"Output directory: {output_dir}")
    print(f"Image features: {features_path} ({feature_size_mb:.2f} MB)")
    print(f"Samples: {samples_path} ({samples_size_mb:.2f} MB)")
    print(f"Total: {feature_size_mb + samples_size_mb:.2f} MB")
    print(f"{'='*50}")
    print(f"Generated {num_samples} samples")
    print(f"Subset size: {subset_size}")
    print(f"Include image features: {include_image_features}")
    print(f"Image size: {img_size}x{img_size}")

    # Compare to old approach
    old_approach_mb = num_samples * 2 * subset_size * sample_feature.numel() * 4 / (1024 * 1024)
    print(f"\nComparison to old approach:")
    print(f"  Old approach (duplicated features): ~{old_approach_mb:.1f} MB")
    print(f"  New approach (shared features): ~{feature_size_mb + samples_size_mb:.1f} MB")
    print(f"  Savings: ~{(1 - (feature_size_mb + samples_size_mb) / old_approach_mb) * 100:.1f}%")

    return dataset


def load_efficient_dataset(
    output_dir: pathlib.Path,
    include_image_features: bool = True
) -> EfficientSfmPairDataset:
    """
    Load a previously generated efficient dataset.

    Args:
        output_dir: Directory containing the generated data
        include_image_features: Whether to include image features in node features

    Returns:
        Loaded EfficientSfmPairDataset
    """
    samples_path = output_dir / 'samples.pt'
    features_path = output_dir / 'image_features.pt'

    return EfficientSfmPairDataset.load(samples_path, features_path, include_image_features)


if __name__ == '__main__':
    # Configure data generation
    model_path = pathlib.Path('./output/01_nerf/02_lego_large/1762043798762/0')
    image_dir = pathlib.Path('data', '01_nerf', '02_lego_large')
    output_dir = pathlib.Path('data/efficient')

    # Generate data
    dataset = generate_sfm_data_efficient(
        model_path=model_path,
        image_dir=image_dir,
        num_samples=1000,  # Can generate many more samples now!
        subset_size=15,
        include_image_features=True,
        img_size=100,
        translation_range=(-10.0, 10.0),
        random_seed=42,
        output_dir=output_dir,
    )

    # Test loading
    print("\nTesting dataset loading...")
    loaded_dataset = load_efficient_dataset(output_dir)
    print(f"Loaded dataset with {len(loaded_dataset)} samples")

    # Test getting a sample
    graph_a, graph_b, label_trans, label_quat = loaded_dataset[0]
    print(f"Sample 0:")
    print(f"  Graph A: {graph_a.x.shape} nodes, {graph_a.edge_index.shape[1]} edges")
    print(f"  Graph B: {graph_b.x.shape} nodes, {graph_b.edge_index.shape[1]} edges")
    print(f"  Label translation: {label_trans}")
    print(f"  Label quaternion: {label_quat}")

