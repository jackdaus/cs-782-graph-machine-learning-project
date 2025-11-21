from typing import List, Optional
import warnings
from pickle import UnpicklingError

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import DataEdgeAttr

try:
    from torch.serialization import add_safe_globals
except ImportError:  # older torch versions
    add_safe_globals = None

class GraphPairDataset(Dataset):
    g_list_1: List[Data]  # Declare the attribute with its type
    g_list_2: List[Data]
    labels: List[torch.Tensor]

    def __init__(self, g_list_1, g_list_2, label_list):
        assert(len(g_list_1) == len(g_list_2) and len(g_list_1) == len(label_list))
        super().__init__()
        self.g_list_1 = g_list_1
        self.g_list_2 = g_list_2
        self.label_list = label_list

    def len(self):
        return len(self.label_list)

    def get(self, idx):
        # Just return the tuple
        data_1 = self.g_list_1[idx]
        data_2 = self.g_list_2[idx]
        label = self.label_list[idx]
        return data_1, data_2, label

    def save(self, path: str) -> None:
        """Persist the dataset to disk with torch.save so it can be reloaded."""
        payload = {
            "g_list_1": self.g_list_1,
            "g_list_2": self.g_list_2,
            "label_list": self.label_list,
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str, map_location: Optional[str] = None) -> "GraphPairDataset":
        """Restore a dataset saved with `save`, handling PyTorch >=2.6 safe loading."""
        payload = GraphPairDataset._safe_torch_load(path, map_location)
        return GraphPairDataset(
            payload["g_list_1"],
            payload["g_list_2"],
            payload["label_list"],
        )

    @staticmethod
    def _safe_torch_load(path: str, map_location: Optional[str] = None):
        # Allowlist DataEdgeAttr so torch.load can unpickle PyG graphs under weights_only=True
        if add_safe_globals is not None:
            add_safe_globals([DataEdgeAttr])
        try:
            return torch.load(path, map_location=map_location)
        except (RuntimeError, UnpicklingError) as exc:
            if "weights_only" not in str(exc):
                raise
            warnings.warn(
                "GraphPairDataset.load falling back to weights_only=False; only do this with trusted files.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.load(path, map_location=map_location, weights_only=False)
