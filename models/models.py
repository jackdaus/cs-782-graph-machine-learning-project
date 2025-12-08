import roma
import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool, GATv2Conv
import torch.nn.functional as F
import torch_geometric.data

class GraphEmbeddingGCN_v1(nn.Module):
    """
    A simple GCN model that produces a graph-level embedding.
    """
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

class GraphEmbeddingGCN_v2(nn.Module):
    """
    A simple GCN model that produces a graph-level embedding. Slightly larger than v1.
    """
    def __init__(self, num_node_features: int, output_dim: int):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.linear = nn.Linear(128, output_dim)

    def forward(self, graph_data: torch_geometric.data.Data):
        h = self.conv1(graph_data.x, graph_data.edge_index)
        h = F.relu(h)
        h = self.conv2(h, graph_data.edge_index)
        h = F.relu(h)
        # Pool all the node features to get a graph-level representation
        h = global_mean_pool(h, graph_data.batch)
        h = self.linear(h)
        return h

class GraphEmbeddingGAT_v3(nn.Module):
    """
    A GAT-based network for embedding a graph into a representation space.
    """
    def __init__(self, num_node_features: int, output_dim: int):
        super().__init__()
        self.conv1 = GATv2Conv(num_node_features, 128)
        self.conv2 = GATv2Conv(128, 128)
        self.linear = nn.Linear(128, output_dim)

    def forward(self, graph_data: torch_geometric.data.Data):
        h = self.conv1(graph_data.x, graph_data.edge_index)
        h = F.relu(h)
        h = self.conv2(h, graph_data.edge_index)
        h = F.relu(h)
        # Pool all the node features to get a graph-level representation
        h = global_mean_pool(h, graph_data.batch)
        h = self.linear(h)
        return h

# This is a super simple network that takes a graph as input, and outputs 3 scalars (x, y, z translation)
class SiameseGCN_v1(nn.Module):
    """A Siamese GCN model that predicts (x, y, z) translation between two graphs."""
    def __init__(self, num_node_features: int):
        super().__init__()
        # Create twin branches that process each graph in tandem
        graph_embedding_dim = 4
        self.sisterA = GraphEmbeddingGCN_v1(num_node_features, output_dim = graph_embedding_dim)
        self.sisterB = GraphEmbeddingGCN_v1(num_node_features, output_dim = graph_embedding_dim)

        # This next part will take as input the embeddings from the sister network
        # self.conv1 = GCNConv(graph_embedding_dim * 2, 4)
        # TODO these linear layers are not used, remove?
        # self.linear1 = nn.Linear(graph_embedding_dim * 2, 8)
        # self.linear2 = nn.Linear(graph_embedding_dim * 2, 8)
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


class SiameseGCN_v2(nn.Module):
    """A Siamese GCN model that predicts (x, y, z) translation and (x, y, z, q) quat rotation between two graphs."""
    def __init__(self, num_node_features: int, graph_embedding_dim: int = 4):
        super().__init__()
        # Create twin branches that process each graph in tandem
        self.sisterA = GraphEmbeddingGCN_v1(num_node_features, output_dim = graph_embedding_dim)
        self.sisterB = GraphEmbeddingGCN_v1(num_node_features, output_dim = graph_embedding_dim)

        # This next part will take as input the embeddings from the sister network
        self.mlp_translation = nn.Sequential(
            nn.Linear(graph_embedding_dim * 2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            # Output 3 scalars for (x, y, z) translation prediction
            nn.Linear(8, 3)
        )
        self.mlp_rotation = nn.Sequential(
            nn.Linear(graph_embedding_dim * 2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            # Output 4 scalars for (x, y, z, w) quat rotation prediction
            nn.Linear(8, 4)
        )

    def forward(self, graph_data_1: torch_geometric.data.Data, graph_data_2: torch_geometric.data.Data) -> tuple[torch.Tensor, torch.Tensor]:
        # Project graph1 and graph2 into the "embedding space" (idk if this is properly called an embedding space...)
        e1 = self.sisterA(graph_data_1)
        e2 = self.sisterB(graph_data_2)

        # Concatenate the features
        h = torch.cat((e1, e2), dim=1)
        # Run through an MLP to predict translation
        h_translation = self.mlp_translation(h)
        h_rotation = self.mlp_rotation(h)
        # Join both outputs into a tuple
        h = (h_translation, h_rotation)
        return h



class SiameseGCN_v3(nn.Module):
    """A Siamese GCN model that predicts a translation (x, y, z) and rotation (3x3 mat) between two graphs."""
    def __init__(self, num_node_features: int, graph_embedding_dim: int = 4):
        super().__init__()
        # Create twin branches that process each graph in tandem
        self.sisterA = GraphEmbeddingGCN_v2(num_node_features, output_dim = graph_embedding_dim)
        self.sisterB = GraphEmbeddingGCN_v2(num_node_features, output_dim = graph_embedding_dim)

        # This next part will take as input the embeddings from the sister network
        self.mlp_translation = nn.Sequential(
            nn.Linear(graph_embedding_dim * 2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            # Output 3 scalars for (x, y, z) translation prediction
            nn.Linear(8, 3)
        )
        self.mlp_rotation = nn.Sequential(
            nn.Linear(graph_embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # Output 9 scalars for a (3x3) rotation matrix prediction
            nn.Linear(32, 9)
        )

    def forward(self, graph_data_1: torch_geometric.data.Data, graph_data_2: torch_geometric.data.Data) -> tuple[torch.Tensor, torch.Tensor]:
        # Project graph1 and graph2 into the "embedding space" (idk if this is properly called an embedding space...)
        e1 = self.sisterA(graph_data_1)
        e2 = self.sisterB(graph_data_2)

        # Concatenate the features
        h = torch.cat((e1, e2), dim=1)
        # Run through an MLP to predict translation
        h_translation = self.mlp_translation(h)
        # Run through an MLP to predict rotation matrix (as a flattened 9-vector)
        rot_9vec = self.mlp_rotation(h)
        # We need to reshape this to (batch_size, 3, 3)
        rot_mat_raw = rot_9vec.view(-1, 3, 3)
        # We need to ensure this matrix is valid rotation matrix (orthogonal and det = 1)
        rot_mat = roma.special_procrustes(rot_mat_raw)
        # Return both outputs as a tuple
        h = (h_translation, rot_mat)
        return h


class SiameseGCN_v4(nn.Module):
    """
    A Siamese GCN model that predicts a translation (x, y, z) and rotation (3x3 mat) between two graphs.
    This version does not normalize the rotation matrix output.
    """
    def __init__(self, num_node_features: int, graph_embedding_dim: int = 4):
        super().__init__()
        # Create twin branches that process each graph in tandem
        self.sisterA = GraphEmbeddingGCN_v2(num_node_features, output_dim = graph_embedding_dim)
        self.sisterB = GraphEmbeddingGCN_v2(num_node_features, output_dim = graph_embedding_dim)

        # This next part will take as input the embeddings from the sister network
        self.mlp_translation = nn.Sequential(
            nn.Linear(graph_embedding_dim * 2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            # Output 3 scalars for (x, y, z) translation prediction
            nn.Linear(8, 3)
        )
        self.mlp_rotation = nn.Sequential(
            nn.Linear(graph_embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # Output 9 scalars for a (3x3) rotation matrix prediction
            nn.Linear(32, 9)
        )

    def forward(self, graph_data_1: torch_geometric.data.Data, graph_data_2: torch_geometric.data.Data) -> tuple[torch.Tensor, torch.Tensor]:
        # Project graph1 and graph2 into the "embedding space" (idk if this is properly called an embedding space...)
        e1 = self.sisterA(graph_data_1)
        e2 = self.sisterB(graph_data_2)

        # Concatenate the features
        h = torch.cat((e1, e2), dim=1)
        # Run through an MLP to predict translation
        h_translation = self.mlp_translation(h)
        # Run through an MLP to predict rotation matrix (as a flattened 9-vector)
        rot_9vec = self.mlp_rotation(h)
        # We need to reshape this to (batch_size, 3, 3)
        rot_mat_raw = rot_9vec.view(-1, 3, 3)
        # We need to ensure this matrix is valid rotation matrix (orthogonal and det = 1)
        # rot_mat = roma.special_procrustes(rot_mat_raw)
        # Return both outputs as a tuple
        h = (h_translation, rot_mat_raw)
        return h



class SiameseGCN_v5(nn.Module):
    """
    A Siamese GCN model that predicts a translation (x, y, z) and rotation (3x3 mat) between two graphs.
    This version does not normalize the rotation matrix output.
    This version uses only one sister network for both inputs (weight sharing).
    """
    def __init__(self, num_node_features: int, graph_embedding_dim: int = 4):
        super().__init__()
        # Use only one sister network for both inputs (weight sharing)
        self.sfm_encoder = GraphEmbeddingGCN_v2(num_node_features, output_dim = graph_embedding_dim)

        # This next part will take as input the embeddings from the sister network
        self.mlp_translation = nn.Sequential(
            nn.Linear(graph_embedding_dim * 2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            # Output 3 scalars for (x, y, z) translation prediction
            nn.Linear(8, 3)
        )
        self.mlp_rotation = nn.Sequential(
            nn.Linear(graph_embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # Output 9 scalars for a (3x3) rotation matrix prediction
            nn.Linear(32, 9)
        )

    def forward(self, graph_data_1: torch_geometric.data.Data, graph_data_2: torch_geometric.data.Data) -> tuple[torch.Tensor, torch.Tensor]:
        # Project graph1 and graph2 into the "embedding space" (idk if this is properly called an embedding space...)
        e1 = self.sfm_encoder(graph_data_1)
        e2 = self.sfm_encoder(graph_data_2)

        # Concatenate the features
        h = torch.cat((e1, e2), dim=1)
        # Run through an MLP to predict translation
        h_translation = self.mlp_translation(h)
        # Run through an MLP to predict rotation matrix (as a flattened 9-vector)
        rot_9vec = self.mlp_rotation(h)
        # We need to reshape this to (batch_size, 3, 3)
        rot_mat_raw = rot_9vec.view(-1, 3, 3)
        # We need to ensure this matrix is valid rotation matrix (orthogonal and det = 1)
        # rot_mat = roma.special_procrustes(rot_mat_raw)
        # Return both outputs as a tuple
        h = (h_translation, rot_mat_raw)
        return h



class SiameseGCN_v6(nn.Module):
    """
    A Siamese GCN model that predicts a translation (x, y, z) and rotation (3x3 mat) between two graphs.
    This version does not normalize the rotation matrix output.
    This uses weight sharing between the two sister graph embedding networks.

    """
    def __init__(self, num_node_features: int, graph_embedding_dim: int = 4):
        super().__init__()
        # Use only one sister network for both inputs (weight sharing)
        self.sfm_encoder = GraphEmbeddingGCN_v2(num_node_features, output_dim = graph_embedding_dim)

        # This next part will take as input the embeddings from the sister network
        self.mlp_trans_rot_head = nn.Sequential(
            nn.Linear(graph_embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # Output 12 scalars: 3 scalars for the translation (x, y, z) and 9 scalars for a (3x3) rotation matrix prediction
            nn.Linear(32, 12)
        )

    def forward(self, graph_data_1: torch_geometric.data.Data, graph_data_2: torch_geometric.data.Data) -> tuple[torch.Tensor, torch.Tensor]:
        # Project graph1 and graph2 into the "embedding space" (idk if this is properly called an embedding space...)
        e1 = self.sfm_encoder(graph_data_1)
        e2 = self.sfm_encoder(graph_data_2)

        # Concatenate the features
        h = torch.cat((e1, e2), dim=1)
        # Run through an MLP to predict translation and rotation together
        h_trans_rot = self.mlp_trans_rot_head(h)
        # The translation is the first 3 values
        h_translation = h_trans_rot[:, :3]
        # The rotation is the last 9 values
        h_rot_9vec = h_trans_rot[:, 3:]
        # Reshape the rotation part to (batch_size, 3, 3)
        rot_mat_raw = h_rot_9vec.view(-1, 3, 3)
        # Return both outputs as a tuple
        h = (h_translation, rot_mat_raw)
        return h


class SiameseGAT_v7(nn.Module):
    """
    A Siamese GCN model that predicts a translation (x, y, z) and rotation (3x3 mat) between two graphs.
    This version does not normalize the rotation matrix output.
    This uses weight sharing between the two sister graph embedding networks.

    """
    def __init__(self, num_node_features: int, graph_embedding_dim: int = 4):
        super().__init__()
        # Use only one sister network for both inputs (weight sharing)
        self.sfm_encoder = GraphEmbeddingGAT_v3(num_node_features, output_dim = graph_embedding_dim)

        # This next part will take as input the embeddings from the sister network
        self.mlp_trans_rot_head = nn.Sequential(
            nn.Linear(graph_embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # Output 12 scalars: 3 scalars for the translation (x, y, z) and 9 scalars for a (3x3) rotation matrix prediction
            nn.Linear(32, 12)
        )

    def forward(self, graph_data_1: torch_geometric.data.Data, graph_data_2: torch_geometric.data.Data) -> tuple[torch.Tensor, torch.Tensor]:
        # Project graph1 and graph2 into the "embedding space" (idk if this is properly called an embedding space...)
        e1 = self.sfm_encoder(graph_data_1)
        e2 = self.sfm_encoder(graph_data_2)

        # Concatenate the features
        h = torch.cat((e1, e2), dim=1)
        # Run through an MLP to predict translation and rotation together
        h_trans_rot = self.mlp_trans_rot_head(h)
        # The translation is the first 3 values
        h_translation = h_trans_rot[:, :3]
        # The rotation is the last 9 values
        h_rot_9vec = h_trans_rot[:, 3:]
        # Reshape the rotation part to (batch_size, 3, 3)
        rot_mat_raw = h_rot_9vec.view(-1, 3, 3)
        # Return both outputs as a tuple
        h = (h_translation, rot_mat_raw)
        return h