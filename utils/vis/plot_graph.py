import networkx
from matplotlib import pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
# from torch_geometric.data import Data
import torch_geometric
import plotly.graph_objs as go
import numpy as np

def plot_networkx(data: torch_geometric.data.Data):
    graph = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(5,5))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=42), with_labels=True, width=0.1)
    plt.show()


def plot_3D_graph(graph_a: torch_geometric.data.Data, graph_b: torch_geometric.data.Data):
    # TODO consider using an intermediary class like sfmNetwork to hold the data more cleanly. (Maybe extend torch_geometric.data.Data))
    def _to_numpy(arr):
        try:
            import torch
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(arr)

    def _edge_trace(centers_np: np.ndarray, edge_index, name: str, color: str):
        if edge_index is None:
            return None
        idx = _to_numpy(edge_index)
        if idx.size == 0:
            return None

        # Normalize shape to \[E, 2]
        if idx.ndim == 2 and idx.shape[0] == 2:
            src, dst = idx[0], idx[1]
            pairs = np.stack([src, dst], axis=1)
        else:
            pairs = idx

        # Unique undirected edges
        u = np.minimum(pairs[:, 0], pairs[:, 1])
        v = np.maximum(pairs[:, 0], pairs[:, 1])
        uv = np.stack([u, v], axis=1)
        uv = np.unique(uv, axis=0)

        xs, ys, zs = [], [], []
        for a, b in uv:
            xs += [centers_np[a, 0], centers_np[b, 0], None]
            ys += [centers_np[a, 1], centers_np[b, 1], None]
            zs += [centers_np[a, 2], centers_np[b, 2], None]

        return go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            # name=name,
            line=dict(color=color, width=1),
            opacity=0.3,
            hoverinfo='skip',
            showlegend=False,
        )

    team_a_camera_centers = _to_numpy(graph_a.x)
    team_b_camera_centers = _to_numpy(graph_b.x)

    traces = []
    traces.append(go.Scatter3d(
        x=team_a_camera_centers[:, 0],
        y=team_a_camera_centers[:, 1],
        z=team_a_camera_centers[:, 2],
        mode='markers',
        name='Set A Images',
        marker=dict(size=5, color='red', symbol='diamond', opacity=1.0),
        hovertemplate='%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>'
    ))
    traces.append(go.Scatter3d(
        x=team_b_camera_centers[:, 0],
        y=team_b_camera_centers[:, 1],
        z=team_b_camera_centers[:, 2],
        mode='markers',
        name='Set B Images',
        marker=dict(size=5, color='blue', symbol='diamond', opacity=1.0),
        hovertemplate='%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>'
    ))

    # Edge traces
    edge_trace_a = _edge_trace(team_a_camera_centers, getattr(graph_a, 'edge_index', None), 'Set A Edges', 'rgba(255,0,0,0.6)')
    edge_trace_b = _edge_trace(team_b_camera_centers, getattr(graph_b, 'edge_index', None), 'Set B Edges', 'rgba(0,0,255,0.6)')
    if edge_trace_a: traces.append(edge_trace_a)
    if edge_trace_b: traces.append(edge_trace_b)

    fig = go.Figure(data=traces)

    combined_xyz = np.vstack((team_a_camera_centers, team_b_camera_centers))
    xyz_min = combined_xyz.min(axis=0)
    xyz_max = combined_xyz.max(axis=0)
    xyz_center = (xyz_min + xyz_max) * 0.5
    xyz_half_range = (xyz_max - xyz_min) * 0.5
    cube_radius = float(np.max(xyz_half_range)) or 1.0

    x_range = [xyz_center[0] - cube_radius, xyz_center[0] + cube_radius]
    y_range = [xyz_center[1] - cube_radius, xyz_center[1] + cube_radius]
    z_range = [xyz_center[2] - cube_radius, xyz_center[2] + cube_radius]

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=x_range),
            yaxis=dict(title='Y', range=y_range),
            zaxis=dict(title='Z', range=z_range),
            aspectmode='cube',
        ),
        title='3D Graph Visualization',
        width=500,
        height=500
    )
    fig.show()
