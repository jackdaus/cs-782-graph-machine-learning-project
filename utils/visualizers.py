import numpy as np
import matplotlib.pyplot as plt
from pycolmap import Reconstruction
import plotly.graph_objs as go

def colmap_3d_points(reconstruction: Reconstruction, filter_outliers: bool = False):
    # Extract 3D point coordinates
    points_3d = filter_points(reconstruction, filter_outliers)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
               s=1, c='b', alpha=0.3, depthshade=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points from COLMAP Reconstruction')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


def colmap_3d_points_interactive(reconstruction: Reconstruction, filter_outliers: bool = False):
    points_3d = filter_points(reconstruction, filter_outliers)

    fig = go.Figure(data=[go.Scatter3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points_3d[:, 2],  # Color by Z value for effect
            colorscale='Viridis',
            opacity=0.7
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='Interactive 3D Points from COLMAP Reconstruction',
        width=900,
        height=700
    )
    fig.show()


def filter_points(reconstruction: Reconstruction, filter_outliers: bool = False):
    points_3d = np.array([p.xyz for p in reconstruction.points3D.values()])
    # Option to filter out outliers to improve visualization clarity
    if filter_outliers:
        # 3-sigma outlier removal w.r.t. centroid using Euclidean distance
        centroid = points_3d.mean(axis=0)
        dists = np.linalg.norm(points_3d - centroid, axis=1)
        sigma = dists.std()

        sigma_thresh = 3.0
        if sigma > 0:
            mask = dists <= sigma_thresh * sigma
            kept = int(mask.sum())
            dropped = int(mask.size - kept)
            if kept == 0:
                print("All points flagged as outliers; skipping filtering.")
            else:
                points_3d = points_3d[mask]
                print(f"Filtered outliers: kept {kept}, dropped {dropped}.")
        else:
            print("Zero distance std; skipping filtering.")
    return points_3d