"""
Efficient data loading for SfM graph datasets.

This module provides memory-efficient storage and loading of SfM graph data by:
1. Storing image features once in a shared cache (ImageFeatureCache)
2. Storing samples as lightweight references to image IDs (SfmSample)
3. Building full graph data on-the-fly during training (EfficientSfmPairDataset)

This avoids duplicating image features across thousands of samples.
"""

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pycolmap
import torch
from PIL import Image
from torch_geometric.data import Data, Dataset
from torchvision import transforms

import utils.subset
from utils.loaders.dataset import DataSfm


class ImageFeatureCache:
    """
    Caches image features by image_id, computed once.

    This class loads and processes images from a COLMAP reconstruction,
    storing them as flattened tensors that can be shared across many samples.
    """

    def __init__(
        self,
        reconstruction: pycolmap.Reconstruction,
        image_dir: pathlib.Path,
        img_size: int = 100
    ):
        """
        Initialize the image feature cache.

        Args:
            reconstruction: COLMAP reconstruction containing image metadata
            image_dir: Directory containing the source images
            img_size: Size to resize images to (img_size x img_size)
        """
        self.reconstruction = reconstruction
        self.image_dir = image_dir
        self.img_size = img_size
        self._cache: Dict[int, torch.Tensor] = {}
        self._transform = transforms.ToTensor()

    def get_features(self, image_id: int) -> torch.Tensor:
        """
        Get image features for an image_id, computing if needed.

        Args:
            image_id: The COLMAP image ID

        Returns:
            Flattened tensor of image features (img_size * img_size * 3,)
        """
        if image_id not in self._cache:
            image = self.reconstruction.images[image_id]
            image_path = self.image_dir / image.name
            img = Image.open(image_path).convert("RGB")
            img = img.resize((self.img_size, self.img_size), resample=Image.LANCZOS)
            tensor = self._transform(img).flatten()
            self._cache[image_id] = tensor
        return self._cache[image_id]

    def precompute_all(self) -> "ImageFeatureCache":
        """
        Precompute features for all images in reconstruction.

        Returns:
            self for method chaining
        """
        for image_id in self.reconstruction.images.keys():
            self.get_features(image_id)
        return self

    def get_all_features(self) -> Dict[int, torch.Tensor]:
        """
        Get all cached features.

        Returns:
            Dict mapping image_id to feature tensor
        """
        return self._cache

    def save(self, path: pathlib.Path) -> None:
        """
        Save cached features to disk.

        Args:
            path: Path to save the cache file
        """
        torch.save({
            'features': self._cache,
            'img_size': self.img_size,
        }, path)
        print(f"Saved {len(self._cache)} image features to {path}")

    @classmethod
    def load(cls, path: pathlib.Path) -> Tuple[Dict[int, torch.Tensor], int]:
        """
        Load cached features from disk.

        Args:
            path: Path to the cache file

        Returns:
            Tuple of (features dict, img_size)
        """
        data = torch.load(path, weights_only=True)
        return data['features'], data['img_size']


@dataclass
class SfmSample:
    """
    Lightweight sample storing only image IDs and transformations.

    This stores references to images rather than the image data itself,
    allowing the same image features to be shared across many samples.
    """
    image_ids_a: List[int]
    image_ids_b: List[int]
    edge_index_a: torch.Tensor
    edge_index_b: torch.Tensor
    # Store poses as (quat, translation) tuples for serialization
    poses_a: List[Tuple[np.ndarray, np.ndarray]]  # List of (quat_xyzw, translation)
    poses_b: List[Tuple[np.ndarray, np.ndarray]]  # List of (quat_xyzw, translation)
    label_translation: torch.Tensor
    label_quat_xyzw: torch.Tensor


def sfm_sample_to_data_sfm(sample: SfmSample) -> Tuple[DataSfm, DataSfm]:
    """
    Convert an SfmSample to a pair of DataSfm objects for visualization.

    This is useful when you want to use visualization functions like
    `utils.vis.plot_3D_graph_sfm` which expect DataSfm objects.

    Args:
        sample: An SfmSample containing image IDs, edge indices, and poses

    Returns:
        Tuple of (graph_a, graph_b) as DataSfm objects
    """
    def build_data_sfm(image_ids, edge_index, poses):
        # Convert poses (quat_xyzw, translation) to pycolmap.Rigid3d objects
        rigid3d_list = []
        for quat_xyzw, translation in poses:
            rigid = pycolmap.Rigid3d(quat_xyzw, translation)
            rigid3d_list.append(rigid)

        # Build node features (positions and rotations) for x attribute
        positions = torch.tensor([pose[1] for pose in poses], dtype=torch.float32)
        rotations = torch.tensor([pose[0] for pose in poses], dtype=torch.float32)
        x = torch.cat([positions, rotations], dim=1)

        data = DataSfm(x=x, edge_index=edge_index)
        data.rigid3d = rigid3d_list
        return data

    graph_a = build_data_sfm(sample.image_ids_a, sample.edge_index_a, sample.poses_a)
    graph_b = build_data_sfm(sample.image_ids_b, sample.edge_index_b, sample.poses_b)

    return graph_a, graph_b


class EfficientSfmPairDataset(Dataset):
    """
    Memory-efficient dataset that references shared image features.

    Instead of storing image features per sample, this dataset stores
    lightweight SfmSample objects that reference image IDs. The full
    graph data is constructed on-the-fly during __getitem__ by looking
    up image features from a shared cache.
    """

    def __init__(
        self,
        samples: List[SfmSample],
        image_features: Dict[int, torch.Tensor],
        include_image_features: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            samples: List of SfmSample objects
            image_features: Dict mapping image_id to feature tensor
            include_image_features: Whether to include image features in node features
        """
        super().__init__()
        self.samples = samples
        self.image_features = image_features
        self.include_image_features = include_image_features
        self.metadata: Dict = {}  # Will be populated when loaded from disk

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> Tuple[Data, Data, torch.Tensor, torch.Tensor]:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (graph_a, graph_b, label_translation, label_quat_xyzw)
        """
        sample = self.samples[idx]

        # Build graph A
        graph_a = self._build_graph(
            sample.image_ids_a,
            sample.edge_index_a,
            sample.poses_a
        )

        # Build graph B
        graph_b = self._build_graph(
            sample.image_ids_b,
            sample.edge_index_b,
            sample.poses_b
        )

        return graph_a, graph_b, sample.label_translation, sample.label_quat_xyzw

    def get_as_data_sfm(self, idx: int) -> Tuple[DataSfm, DataSfm, torch.Tensor, torch.Tensor]:
        """
        Get a sample as DataSfm objects for visualization.

        This is useful when you want to use visualization functions like
        `visualize_sfm_prediction` which expect DataSfm objects with rigid3d attributes.

        Args:
            idx: Sample index

        Returns:
            Tuple of (graph_a, graph_b, label_translation, label_quat_xyzw) with DataSfm objects
        """
        sample = self.samples[idx]
        graph_a, graph_b = sfm_sample_to_data_sfm(sample)

        # Add image features to x if enabled
        if self.include_image_features:
            img_feats_a = torch.stack([self.image_features[id] for id in sample.image_ids_a])
            img_feats_b = torch.stack([self.image_features[id] for id in sample.image_ids_b])
            graph_a.x = torch.cat([graph_a.x, img_feats_a], dim=1)
            graph_b.x = torch.cat([graph_b.x, img_feats_b], dim=1)

        return graph_a, graph_b, sample.label_translation, sample.label_quat_xyzw

    def _build_graph(
        self,
        image_ids: List[int],
        edge_index: torch.Tensor,
        poses: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Data:
        """
        Build a PyG Data graph with node features computed on-the-fly.

        Args:
            image_ids: List of image IDs for nodes
            edge_index: Edge index tensor (2, E)
            poses: List of (quat_xyzw, translation) tuples for each node

        Returns:
            PyG Data object with node features and edge index
        """
        # Extract positions and rotations from poses
        # Convert to numpy array first to avoid slow tensor creation from list of arrays
        positions = torch.from_numpy(
            np.array([pose[1] for pose in poses], dtype=np.float32)
        )
        rotations = torch.from_numpy(
            np.array([pose[0] for pose in poses], dtype=np.float32)
        )

        # Build node features
        if self.include_image_features:
            img_feats = torch.stack([self.image_features[id] for id in image_ids])
            x = torch.cat([positions, rotations, img_feats], dim=1)
        else:
            x = torch.cat([positions, rotations], dim=1)

        return Data(x=x, edge_index=edge_index)

    def save(self, path: pathlib.Path, metadata: Optional[Dict] = None) -> None:
        """
        Save dataset samples to disk (features stored separately).

        Args:
            path: Path to save the samples file
            metadata: Optional metadata dict to store alongside samples
        """
        # Convert samples to serializable format
        serializable_samples = []
        for s in self.samples:
            serializable_samples.append({
                'image_ids_a': s.image_ids_a,
                'image_ids_b': s.image_ids_b,
                'edge_index_a': s.edge_index_a,
                'edge_index_b': s.edge_index_b,
                'poses_a': s.poses_a,
                'poses_b': s.poses_b,
                'label_translation': s.label_translation,
                'label_quat_xyzw': s.label_quat_xyzw,
            })

        torch.save({
            'samples': serializable_samples,
            'include_image_features': self.include_image_features,
            'metadata': metadata or {},
        }, path)
        print(f"Saved {len(self.samples)} samples to {path}")

    @classmethod
    def load(
        cls,
        samples_path: pathlib.Path,
        features_path: pathlib.Path,
        include_image_features: bool = True
    ) -> "EfficientSfmPairDataset":
        """
        Load dataset from disk.

        Args:
            samples_path: Path to the samples file
            features_path: Path to the image features cache file
            include_image_features: Whether to include image features in node features

        Returns:
            Loaded EfficientSfmPairDataset with metadata attribute
        """
        data = torch.load(samples_path, weights_only=False)

        # Only load features if needed
        if include_image_features:
            features, img_size = ImageFeatureCache.load(features_path)
        else:
            features = {}
            img_size = None

        samples = []
        for s in data['samples']:
            samples.append(SfmSample(
                image_ids_a=s['image_ids_a'],
                image_ids_b=s['image_ids_b'],
                edge_index_a=s['edge_index_a'],
                edge_index_b=s['edge_index_b'],
                poses_a=s['poses_a'],
                poses_b=s['poses_b'],
                label_translation=s['label_translation'],
                label_quat_xyzw=s['label_quat_xyzw'],
            ))

        dataset = cls(samples, features, include_image_features)
        dataset.metadata = data.get('metadata', {})
        if img_size is not None:
            dataset.metadata['img_size'] = img_size
        return dataset


def compute_edge_index(
    reconstruction: pycolmap.Reconstruction,
    image_ids: List[int]
) -> torch.Tensor:
    """
    Compute edge index based on covisibility between images.

    Args:
        reconstruction: COLMAP reconstruction
        image_ids: List of image IDs to include

    Returns:
        Edge index tensor of shape (2, E)
    """
    image_id_to_idx = {image_id: idx for idx, image_id in enumerate(image_ids)}

    src, dst = [], []
    for src_id in image_ids:
        for dst_id in utils.subset.get_covisible_image_ids(reconstruction, src_id):
            if dst_id in image_id_to_idx:
                src.append(image_id_to_idx[src_id])
                dst.append(image_id_to_idx[dst_id])

    return torch.tensor([src, dst], dtype=torch.long)


