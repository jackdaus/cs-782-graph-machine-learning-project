"""
Generate synthetic SfM training data by applying random transformations to COLMAP reconstructions.

This script:
1. Loads a COLMAP reconstruction
2. Creates random subsets of images
3. Applies random rotations and translations to one subset
4. Saves the generated graph pairs and labels to disk
"""

import pathlib
import random
import numpy as np
import torch
import pycolmap
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import utils.subset
import utils.loaders


def generate_sfm_data(
    model_path,
    image_dir,
    num_samples=10,
    subset_size=15,
    include_image_features=False,
    translation_range=(-10.0, 10.0),
    random_seed=42,
    output_path=None
):
    """
    Generate synthetic SfM training data.

    Args:
        model_path: Path to COLMAP reconstruction model
        image_dir: Path to source images directory
        num_samples: Number of data samples to generate
        subset_size: Size of each image subset
        include_image_features: Whether to include image features in node features
        translation_range: Range for random translations (min, max)
        random_seed: Random seed for reproducibility
        output_path: Path to save the generated data (default: data/data.pt)

    Returns:
        Dictionary containing generated data
    """
    # Set random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load the Colmap data
    reconstruction = pycolmap.Reconstruction(model_path)

    # Initialize lists to store data
    graph_a_list = []
    graph_b_list = []
    labels_translations = []
    labels_quat_xyzw = []

    # Generate data samples
    for i in tqdm(range(num_samples), desc="Generating data"):
        # Create random subsets using distance-based heuristic
        a_ids, b_ids = utils.subset.create_image_subsets_from_distance(
            reconstruction, subset_size, random_captains=True
        )

        # Create PyG graph Dataset for each subset
        graph_a = utils.loaders.load_reconstruction_to_graph_sfm(
            reconstruction, image_dir, a_ids, include_image_features=include_image_features
        )
        graph_b = utils.loaders.load_reconstruction_to_graph_sfm(
            reconstruction, image_dir, b_ids, include_image_features=include_image_features
        )

        # Generate random translation
        random_translation = torch.empty(3).uniform_(*translation_range)
        # random_translation = torch.tensor([0, 0, 0]).float()  # Zero translation for now

        # Generate random rotation
        random_rotation = Rotation.random().as_quat()

        # Create Rigid3D transformation
        random_transform = pycolmap.Rigid3d(random_rotation, random_translation)

        # Apply the rotation/translation to graph_b
        graph_b.rigid3d = [random_transform * rigid for rigid in graph_b.rigid3d]

        # Update node features with the transformed values
        graph_b_pos = np.stack([rigid.translation for rigid in graph_b.rigid3d], axis=0)
        graph_b_pos = torch.from_numpy(graph_b_pos).float()

        graph_b_rot = np.stack([rigid.rotation.quat for rigid in graph_b.rigid3d], axis=0)
        graph_b_rot = torch.from_numpy(graph_b_rot).float()

        # Concatenate features
        if include_image_features:
            graph_b.x = torch.hstack([graph_b_pos, graph_b_rot, graph_b.image_features])
        else:
            graph_b.x = torch.hstack([graph_b_pos, graph_b_rot])

        # Store the data
        graph_a_list.append(graph_a)
        graph_b_list.append(graph_b)
        labels_translations.append(random_translation)
        random_rotation_tensor = torch.from_numpy(np.stack(random_rotation)).float()
        labels_quat_xyzw.append(random_rotation_tensor)

    # Prepare data dictionary
    data = {
        'graph_a_list': graph_a_list,
        'graph_b_list': graph_b_list,
        'labels_translations': labels_translations,
        'labels_quat_xyzw': labels_quat_xyzw,
        'metadata': {
            'num_samples': num_samples,
            'subset_size': subset_size,
            'include_image_features': include_image_features,
            'translation_range': translation_range,
            'random_seed': random_seed,
            'model_path': str(model_path),
            'image_dir': str(image_dir),
        }
    }

    # Save to disk
    if output_path is None:
        output_path = pathlib.Path('data/data.pt')
    else:
        output_path = pathlib.Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    print(f"\nData saved to {output_path}")
    print(f"Generated {num_samples} samples")
    print(f"Include image features: {include_image_features}")

    return data


if __name__ == '__main__':
    # Configure data generation
    model_path = pathlib.Path('./output/01_nerf/02_lego_large/1762043798762/0')
    image_dir = pathlib.Path('data', '01_nerf', '02_lego_large')

    # Generate data
    data = generate_sfm_data(
        model_path=model_path,
        image_dir=image_dir,
        num_samples=1000,
        subset_size=15,
        include_image_features=True,
        translation_range=(-1.0, 1.0),
        random_seed=42,
        output_path=pathlib.Path('data/data-4.pt')
    )

    print("\nData generation complete!")
    print(f"Metadata: {data['metadata']}")

