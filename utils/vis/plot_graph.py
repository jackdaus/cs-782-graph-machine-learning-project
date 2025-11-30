from matplotlib import pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import torch_geometric
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.colors
import torch
import pycolmap

from utils.loaders.dataset import DataSfm


def plot_networkx(data: torch_geometric.data.Data):
    graph = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(5,5))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=42), with_labels=True, width=0.1)
    plt.show()

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

def plot_3D_graph_subplots(graph_pairs: list[tuple[torch_geometric.data.Data, torch_geometric.data.Data]], subplot_titles: list[str] = None):
    """
    Plots multiple 3D graphs in a row of subplots.

    Args:
        graph_pairs: A list of tuples, where each tuple contains two graphs (graph_a, graph_b) to be plotted in a subplot.
        subplot_titles: A list of titles for each subplot.
    """
    num_plots = len(graph_pairs)
    if num_plots == 0:
        return

    if subplot_titles is None:
        subplot_titles = [f'Graph Pair {i+1}' for i in range(num_plots)]

    fig = make_subplots(
        rows=1, cols=num_plots,
        specs=[[{'type': 'scene'}] * num_plots],
        subplot_titles=subplot_titles
    )


    for i, (graph_a, graph_b) in enumerate(graph_pairs):
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
            hovertemplate='%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>',
            legendgroup='groupA',
            showlegend=(i==0)
        ))
        traces.append(go.Scatter3d(
            x=team_b_camera_centers[:, 0],
            y=team_b_camera_centers[:, 1],
            z=team_b_camera_centers[:, 2],
            mode='markers',
            name='Set B Images',
            marker=dict(size=5, color='blue', symbol='diamond', opacity=1.0),
            hovertemplate='%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>',
            legendgroup='groupB',
            showlegend=(i==0)
        ))

        # Edge traces
        edge_trace_a = _edge_trace(team_a_camera_centers, getattr(graph_a, 'edge_index', None), 'Set A Edges', 'rgba(255,0,0,0.6)')
        edge_trace_b = _edge_trace(team_b_camera_centers, getattr(graph_b, 'edge_index', None), 'Set B Edges', 'rgba(0,0,255,0.6)')
        if edge_trace_a: traces.append(edge_trace_a)
        if edge_trace_b: traces.append(edge_trace_b)

        for trace in traces:
            fig.add_trace(trace, row=1, col=i+1)

        combined_xyz = np.vstack((team_a_camera_centers, team_b_camera_centers))
        xyz_min = combined_xyz.min(axis=0)
        xyz_max = combined_xyz.max(axis=0)
        xyz_center = (xyz_min + xyz_max) * 0.5
        xyz_half_range = (xyz_max - xyz_min) * 0.5
        cube_radius = float(np.max(xyz_half_range)) or 1.0

        x_range = [xyz_center[0] - cube_radius, xyz_center[0] + cube_radius]
        y_range = [xyz_center[1] - cube_radius, xyz_center[1] + cube_radius]
        z_range = [xyz_center[2] - cube_radius, xyz_center[2] + cube_radius]

        scene_layout = dict(
            xaxis=dict(title='X', range=x_range),
            yaxis=dict(title='Y', range=y_range),
            zaxis=dict(title='Z', range=z_range),
            aspectmode='cube',
        )
        fig.update_scenes(scene_layout, row=1, col=i+1)

    fig.update_layout(
        title_text='3D Graph Visualization',
        width=500 * num_plots,
        height=500
    )
    fig.show()


def plot_3D_graph(graph_a: torch_geometric.data.Data, graph_b: torch_geometric.data.Data, title: str = ''):
    plot_3D_graph_subplots([(graph_a, graph_b)], [title])


def plot_3D_graph_subplots_sfm(graph_pairs: list[tuple['DataSfm', 'DataSfm']], subplot_titles: list[str] = None):
    """
    Plots multiple 3D graphs in a row of subplots using DataSfm objects.

    Args:
        graph_pairs: A list of tuples, where each tuple contains two DataSfm graphs (graph_a, graph_b) to be plotted in a subplot.
        subplot_titles: A list of titles for each subplot.
    """
    num_plots = len(graph_pairs)
    if num_plots == 0:
        return

    if subplot_titles is None:
        subplot_titles = [f'Graph Pair {i+1}' for i in range(num_plots)]

    fig = make_subplots(
        rows=1, cols=num_plots,
        specs=[[{'type': 'scene'}] * num_plots],
        subplot_titles=subplot_titles
    )

    for i, (graph_a, graph_b) in enumerate(graph_pairs):
        # Extract camera centers from Rigid3d objects
        team_a_camera_centers = np.array([rigid.translation for rigid in graph_a.rigid3d])
        team_b_camera_centers = np.array([rigid.translation for rigid in graph_b.rigid3d])

        traces = []
        traces.append(go.Scatter3d(
            x=team_a_camera_centers[:, 0],
            y=team_a_camera_centers[:, 1],
            z=team_a_camera_centers[:, 2],
            mode='markers',
            name='Set A Images',
            marker=dict(size=5, color='red', symbol='diamond', opacity=1.0),
            hovertemplate='%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>',
            legendgroup='groupA',
            showlegend=(i==0)
        ))
        traces.append(go.Scatter3d(
            x=team_b_camera_centers[:, 0],
            y=team_b_camera_centers[:, 1],
            z=team_b_camera_centers[:, 2],
            mode='markers',
            name='Set B Images',
            marker=dict(size=5, color='blue', symbol='diamond', opacity=1.0),
            hovertemplate='%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>',
            legendgroup='groupB',
            showlegend=(i==0)
        ))

        # Edge traces
        edge_trace_a = _edge_trace(team_a_camera_centers, getattr(graph_a, 'edge_index', None), 'Set A Edges', 'rgba(255,0,0,0.6)')
        edge_trace_b = _edge_trace(team_b_camera_centers, getattr(graph_b, 'edge_index', None), 'Set B Edges', 'rgba(0,0,255,0.6)')
        if edge_trace_a: traces.append(edge_trace_a)
        if edge_trace_b: traces.append(edge_trace_b)

        for trace in traces:
            fig.add_trace(trace, row=1, col=i+1)

        combined_xyz = np.vstack((team_a_camera_centers, team_b_camera_centers))
        xyz_min = combined_xyz.min(axis=0)
        xyz_max = combined_xyz.max(axis=0)
        xyz_center = (xyz_min + xyz_max) * 0.5
        xyz_half_range = (xyz_max - xyz_min) * 0.5
        cube_radius = float(np.max(xyz_half_range)) or 1.0

        x_range = [xyz_center[0] - cube_radius, xyz_center[0] + cube_radius]
        y_range = [xyz_center[1] - cube_radius, xyz_center[1] + cube_radius]
        z_range = [xyz_center[2] - cube_radius, xyz_center[2] + cube_radius]

        scene_layout = dict(
            xaxis=dict(title='X', range=x_range),
            yaxis=dict(title='Y', range=y_range),
            zaxis=dict(title='Z', range=z_range),
            aspectmode='cube',
        )
        fig.update_scenes(scene_layout, row=1, col=i+1)

    fig.update_layout(
        title_text='3D Graph Visualization (SfM)',
        width=500 * num_plots,
        height=500
    )
    fig.show()


def plot_3D_graph_sfm(graph_a: 'DataSfm', graph_b: 'DataSfm', title: str = ''):
    """
    Plots a single pair of DataSfm graphs in 3D.

    Args:
        graph_a: First DataSfm graph.
        graph_b: Second DataSfm graph.
        title: Title for the plot.
    """
    plot_3D_graph_subplots_sfm([(graph_a, graph_b)], [title])


def plot_many_3D_graphs(graphs: list[torch_geometric.data.Data], titles: list[str] = None):
    """
    Plots multiple 3D graphs in a single plot, each with a different color.

    Args:
        graphs: A list of graphs to be plotted.
        titles: A list of titles for each graph.
    """
    num_graphs = len(graphs)
    if num_graphs == 0:
        return

    fig = go.Figure()

    colors = plotly.colors.qualitative.Plotly

    all_xyz = []

    for i, graph in enumerate(graphs):
        camera_centers = _to_numpy(graph.x)
        all_xyz.append(camera_centers)

        color = colors[i % len(colors)]
        title = titles[i] if titles and i < len(titles) else f'Graph {i+1}'

        fig.add_trace(go.Scatter3d(
            x=camera_centers[:, 0],
            y=camera_centers[:, 1],
            z=camera_centers[:, 2],
            mode='markers',
            name=title,
            marker=dict(size=5, color=color, symbol='diamond', opacity=1.0),
            hovertemplate='%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>',
        ))

        edge_trace = _edge_trace(camera_centers, getattr(graph, 'edge_index', None), title, color)
        if edge_trace:
            fig.add_trace(edge_trace)

    if not all_xyz:
        return

    combined_xyz = np.vstack(all_xyz)
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
        width=800,
        height=800
    )
    fig.show()


def plot_many_3D_graphs_sfm(graphs: list['DataSfm'], titles: list[str] = None, colors: list[str] = None):
    """
    Plots multiple 3D graphs in a single plot using DataSfm objects, each with a different color.

    Args:
        graphs: A list of DataSfm graphs to be plotted.
        titles: A list of titles for each graph.
        colors: A list of color strings for each graph. If None, uses default Plotly colors.
    """
    num_graphs = len(graphs)
    if num_graphs == 0:
        return

    fig = go.Figure()

    if colors is None:
        colors = plotly.colors.qualitative.Plotly

    all_xyz = []

    for i, graph in enumerate(graphs):
        # Extract camera centers from Rigid3d objects
        camera_centers = np.array([rigid.translation for rigid in graph.rigid3d])
        all_xyz.append(camera_centers)

        color = colors[i % len(colors)]
        title = titles[i] if titles and i < len(titles) else f'Graph {i+1}'

        fig.add_trace(go.Scatter3d(
            x=camera_centers[:, 0],
            y=camera_centers[:, 1],
            z=camera_centers[:, 2],
            mode='markers',
            name=title,
            marker=dict(size=5, color=color, symbol='diamond', opacity=1.0),
            hovertemplate='%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>',
        ))

        edge_trace = _edge_trace(camera_centers, getattr(graph, 'edge_index', None), title, color)
        if edge_trace:
            fig.add_trace(edge_trace)

    if not all_xyz:
        return

    combined_xyz = np.vstack(all_xyz)
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
        title='3D Graph Visualization (SfM)',
        width=800,
        height=800
    )
    fig.show()


def visualize_sfm_prediction(model, sample):
    """
    Visualizes the prediction of a rotation/translation model on an SfM graph pair.

    This function:
    1. Runs the model on a graph pair to predict transformation
    2. Applies the inverse predicted transformation to restore graph_b
    3. Applies the inverse true transformation to show ground truth
    4. Plots all four graphs in order:
       - Graph A (true) - dark green
       - Graph B (true) - light green
       - Graph B (input/perturbed) - red
       - Graph B (prediction) - blue

    Note: This function always uses CPU for inference since visualization requires
    converting tensors to numpy arrays.

    Args:
        model: The trained model that predicts (translation, rotation) from graph pairs
        sample: A tuple containing (graph_a, graph_b, label_translation, label_rotation)

    Returns:
        A tuple of (predicted_graph_b, true_graph_b, graph_a_example, graph_b_example)
    """
    # Move model to CPU and set to eval mode (required for visualization)
    model.to('cpu')
    model.eval()

    # Extract sample components and move to CPU
    graph_a_example = sample[0].to('cpu')
    graph_b_example = sample[1].to('cpu')
    label_trans_example = sample[2].to('cpu')
    label_rot_example = sample[3].to('cpu')

    # Make prediction
    with torch.no_grad():
        pred = model(graph_a_example, graph_b_example)

    pred_trans = pred[0]
    pred_rot = pred[1]

    # Important! We must normalize the predicted quaternion... since the prediction might not be normalized
    pred_rot = pred_rot / pred_rot.norm()

    # Create transformation from prediction
    graph_transformation_pred = pycolmap.Rigid3d(
        pred_rot.cpu().numpy().flatten(),
        pred_trans.cpu().numpy().flatten()
    )

    # Compute the inverse, since this is what we will need to "restore" the pose of the graph
    graph_transformation_pred_inv = graph_transformation_pred.inverse()

    # Create copy for the predicted graph
    predicted_graph_b = graph_b_example.clone()
    # Apply restoring predicted transform
    predicted_graph_b.rigid3d = [graph_transformation_pred_inv * rigid for rigid in predicted_graph_b.rigid3d]
    # Update graph attributes
    predicted_graph_b_pos = np.stack([rigid.translation for rigid in predicted_graph_b.rigid3d], axis=0)
    predicted_graph_b_pos = torch.from_numpy(predicted_graph_b_pos).float()
    predicted_graph_b_rot = np.stack([rigid.rotation.quat for rigid in predicted_graph_b.rigid3d], axis=0)
    predicted_graph_b_rot = torch.from_numpy(predicted_graph_b_rot).float()
    predicted_graph_b.x = torch.hstack([
        predicted_graph_b_pos,
        predicted_graph_b_rot,
        # predicted_graph_b.image_features.to('cpu')
    ])

    # Create transformation for true label
    graph_transformation_true = pycolmap.Rigid3d(
        label_rot_example.cpu().numpy().flatten(),
        label_trans_example.cpu().numpy().flatten()
    )

    graph_transformation_true_inv = graph_transformation_true.inverse()
    # Create copy, and then apply the transform to recover the true labels
    true_graph_b = graph_b_example.clone()
    true_graph_b.rigid3d = [graph_transformation_true_inv * rigid for rigid in true_graph_b.rigid3d]

    # Visualize all four graphs in the desired order with custom colors
    # Order: Graph A (true), Graph B (true), Graph B (input), Graph B (prediction)
    # Colors: Dark Green, Light Green, Red, Blue
    plot_many_3D_graphs_sfm(
        [graph_a_example, true_graph_b, graph_b_example, predicted_graph_b],
        ["Graph A (true)", "Graph B (true)", "Graph B (input)", "Graph B (prediction)"],
        colors=['black', 'limegreen', 'red', 'dodgerblue']
    )

    return predicted_graph_b, true_graph_b, graph_a_example, graph_b_example

