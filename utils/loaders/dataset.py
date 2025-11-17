import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

class GraphPairDataset(Dataset):
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