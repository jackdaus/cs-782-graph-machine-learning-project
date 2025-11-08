import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import torch_geometric.data

# This is a super simple network that takes a graph as input, and outputs a single scalar
class GraphEmbeddingGCN_v1(nn.Module):
    def __init__(self, num_node_features: int, output_dim: int):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.linear = nn.Linear(4, output_dim)

    def forward(self, graph_data: torch_geometric.data.Data):
        h = self.conv1(graph_data.x, graph_data.edge_index)
        h = F.relu(h)
        h = self.conv2(h, graph_data.edge_index)
        h = F.relu(h)
        # Pool all the node features to get a graph-level representation
        h = global_mean_pool(h, graph_data.batch)
        h = self.linear(h)
        return h


# This is a super simple network that takes a graph as input, and outputs a single scalar
class SiameseGCN_v1(nn.Module):
    def __init__(self, num_node_features: int):
        super().__init__()
        # Create twin branches that process each graph in tandem
        graph_embedding_dim = 4
        self.sisterA = GraphEmbeddingGCN_v1(num_node_features, output_dim = graph_embedding_dim)
        self.sisterB = GraphEmbeddingGCN_v1(num_node_features, output_dim = graph_embedding_dim)

        # This next part will take as input the embeddings from the sister network
        # self.conv1 = GCNConv(graph_embedding_dim * 2, 4)
        self.linear1 = nn.Linear(graph_embedding_dim * 2, 8)
        self.linear2 = nn.Linear(graph_embedding_dim * 2, 8)
        self.mlp = nn.Sequential(
            nn.Linear(graph_embedding_dim * 2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            # Output 3 scalars for (x, y, z) translation prediction
            nn.Linear(8, 3)
        )

    def forward(self, graph_data_1: torch_geometric.data.Data, graph_data_2: torch_geometric.data.Data):
        # Project graph1 and graph2 into the "embedding space" (idk if this is properly called an embedding space...)
        e1 = self.sisterA(graph_data_1)
        e2 = self.sisterB(graph_data_2)

        # Concatenate the features
        h = torch.cat((e1, e2), dim=1)
        # Run through an MLP to predict 3
        h = self.mlp(h)
        return h