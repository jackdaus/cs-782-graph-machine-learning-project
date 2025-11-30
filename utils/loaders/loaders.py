import pathlib

from pycolmap import Reconstruction
import random
import utils.subset
import torch
from torch_geometric.data import Data
import numpy as np
from PIL import Image
from torchvision import transforms

from utils.loaders.dataset import DataSfm


def load_reconstruction_to_graph_sfm(reconstruction: Reconstruction, image_dir: pathlib.Path,
                                     filtered_image_ids: list[int] = None, img_size: int = 3,
                                     include_image_features: bool = True) -> DataSfm:
    """Convert a COLMAP reconstruction to a PyTorch Geometric graph with SfM data.

    Args:
        reconstruction: COLMAP reconstruction containing images and camera poses
        image_dir: Directory containing the image files
        filtered_image_ids: Subset of image IDs to include (None = all images)
        img_size: Resize dimension for images (default: 3x3), for image feature vector
        include_image_features: Whether to include image feature vectors (default: True)

    Returns:
        DataSfm: Graph with node features (positions, rotations, optionally image features) and edges based on covisibility
    """
    # Get all the image_ids from this Reconstruction
    all_image_ids = list(reconstruction.images.keys())

    # If no image_ids were provided, then we use all images in the Reconstruction
    if filtered_image_ids is None:
        filtered_image_ids = all_image_ids

    assert set(filtered_image_ids).issubset(set(all_image_ids))

    # 1. Get image projection centers
    image_centers = [reconstruction.images[image_id].projection_center() for image_id in filtered_image_ids]
    image_centers_tensor = torch.from_numpy(np.array(image_centers)).float()

    # 2. Rotations
    world_from_cams = [reconstruction.images[image_id].cam_from_world().inverse() for image_id in filtered_image_ids]
    quats_xyzw = [rigid_3d.rotation.quat for rigid_3d in world_from_cams]
    quats_xyzw_tensor = torch.from_numpy(np.array(quats_xyzw)).float()

    # 3. Load images as tensors
    # NOTE: this is a "naive" approach of including image data. We will want to use a CNN
    # or pre-processed image features in our next experiment.
    image_paths = []
    image_files = []
    image_tensors = []
    image_features = None

    if include_image_features:
        transform = transforms.ToTensor()
        for image_id in filtered_image_ids:
            image = reconstruction.images[image_id]
            image_path = image_dir / image.name
            img = Image.open(image_path).convert("RGB")
            img = img.resize((img_size, img_size), resample=Image.LANCZOS) # Resize to 3x3 for now as a quick experiment
            image_paths.append(image_path)
            image_files.append(img)
            image_tensors.append(transform(img))

        # Stack and flatten image tensors
        image_features = torch.stack(image_tensors)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten to (num_nodes, num_features)

    # Create a map between the image_id and index in the list
    image_id_to_idx = {image_id: idx for idx, image_id in enumerate(filtered_image_ids)}

    # Initialize our lists for the edges. Source node and destination node.
    # (The graph is undirected, so "src" and "dst" labels are just for convention here.)
    src, dst = [], []
    # For each image in set A...
    for src_id in filtered_image_ids:
        # Get all other images that have tracks linked to the source image
        for dst_id in utils.subset.get_covisible_image_ids(reconstruction, src_id):
            # We want to make sure the
            if dst_id in image_id_to_idx:
                src.append(image_id_to_idx[src_id])
                dst.append(image_id_to_idx[dst_id])
    # Create the edge index matrix. PyG expects a matrix of shape (2, E)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Build node features: concatenate position, rotation, and optionally image features
    if include_image_features:
        node_features = torch.cat([image_centers_tensor, quats_xyzw_tensor, image_features], dim=1)
    else:
        node_features = torch.cat([image_centers_tensor, quats_xyzw_tensor], dim=1)

    data_sfm = DataSfm(
        # for the node features, we concatenate:
        # 1. image centers (3 dims)
        # 2. quaternions (4 dims)
        # 3. image features (optional, e.g., 27 dims for 3x3 RGB images)
        x=node_features,
        edge_index=edge_index)
    # SfM-specific attributes
    data_sfm.rigid3d = world_from_cams

    # data_sfm.pos = image_centers_tensor
    # data_sfm.quat_xyzw = quats_xyzw_tensor
    # data_sfm.image_tensors = image_tensors
    data_sfm.image_features = image_features
    # Extra info that are not features but might be useful for debug (e.g., image file paths)
    data_sfm.image_paths = image_paths
    return data_sfm


def load_reconstruction_to_graph(reconstruction: Reconstruction, image_dir: pathlib.Path, filtered_image_ids: list[int] = None) -> Data:
    # Get all the image_ids from this Reconstruction
    all_image_ids = list(reconstruction.images.keys())

    # If no image_ids were provided, then we use all images in the Reconstruction
    if filtered_image_ids is None:
        filtered_image_ids = all_image_ids  # TODO make sure this not a reference type!

    assert set(filtered_image_ids).issubset(set(all_image_ids))

    # Calculate per node features
    # image_centers = [image.projection_center() for image in reconstruction.images.values() if
    #                  image.image_id in filtered_image_ids]
    # x = torch.tensor(image_centers, dtype=torch.float)
    # x = torch.from_numpy(np.array(image_centers)).float()

    # 1. Get image projection centers
    image_centers = [reconstruction.images[image_id].projection_center() for image_id in filtered_image_ids]
    image_centers_tensor = torch.from_numpy(np.array(image_centers)).float()


    # 2. Load images as tensors
    # NOTE: this is a "naive" approach of including image data. We will want to use a CNN
    # or pre-processed image features in our next experiment.
    image_paths = []
    image_files = []
    image_tensors = []
    transform = transforms.ToTensor()
    for image_id in filtered_image_ids:
        image = reconstruction.images[image_id]
        image_path = image_dir / image.name
        img = Image.open(image_path).convert("RGB")
        img = img.resize((3,3), resample=Image.LANCZOS) # Resize to 3x3 for now as a quick experiment
        image_paths.append(image_path)
        image_files.append(img)
        image_tensors.append(transform(img))

    # # Stack and flatten image tensors
    image_features = torch.stack(image_tensors)
    image_features = image_features.view(image_features.size(0), -1)  # Flatten to (num_nodes, num_features)

    # # 3. Concatenate both features to create the final node feature matrix 'x'
    x = torch.cat([image_centers_tensor, image_features], dim=1)

    # Create a map between the image_id and index in the list
    image_id_to_idx = {image_id: idx for idx, image_id in enumerate(filtered_image_ids)}

    # Initialize our lists for the edges. Source node and destination node.
    # (The graph is undirected, so "src" and "dst" labels are just for convention here.)
    src, dst = [], []
    # For each image in set A...
    for src_id in filtered_image_ids:
        # Get all other images that have tracks linked to the source image
        for dst_id in utils.subset.get_covisible_image_ids(reconstruction, src_id):
            # We want to make sure the
            if dst_id in image_id_to_idx:
                src.append(image_id_to_idx[src_id])
                dst.append(image_id_to_idx[dst_id])
    # Create the edge index matrix. PyG expects a matrix of shape (2, E)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    # Extra info that might be useful! Might be worth encapsulating this in a defined property to stay organized...
    data.image_tensors = image_tensors
    data.image_paths = image_paths
    data.image_files = image_files
    return data