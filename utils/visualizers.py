import numpy as np
import matplotlib.pyplot as plt
from pycolmap import Reconstruction
import plotly.graph_objs as go

def colmap_3d_points(reconstruction: Reconstruction, filter_outliers: bool = False, show_cameras: bool = False):
    # Extract 3D point coordinates
    points_3d = filter_points(reconstruction, filter_outliers)

    # Extract camera positions
    camera_translations = get_camera_translations(reconstruction)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
        s=1, c='b', alpha=0.3, depthshade=False)
    if show_cameras:
        ax.scatter(camera_translations[:,0], camera_translations[:,1], camera_translations[:,2],
               c='r', marker='*', s=25)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points from COLMAP Reconstruction')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


def colmap_3d_points_interactive(reconstruction: Reconstruction, filter_outliers: bool = False, show_cameras: bool = False):
    # Extract 3D point coordinates
    points_3d = filter_points(reconstruction, filter_outliers)

    # Extract camera positions
    camera_translations = get_camera_translations(reconstruction)
    camera_names = get_camera_filenames(reconstruction)

    traces = [go.Scatter3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points_3d[:, 2],  # Color by Z value for effect
            colorscale='Viridis',
            opacity=0.7
        ),
        hovertemplate='3d point: %{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>'
    )]

    if show_cameras:
        traces.append(go.Scatter3d(
            x=camera_translations[:, 0],
            y=camera_translations[:, 1],
            z=camera_translations[:, 2],
            mode='markers',
            name='cameras',
            text=camera_names,  # list of strings used in hovertemplate
            marker=dict(
                size=5,
                color=camera_translations[:, 2],
                symbol='diamond',
                opacity=1.0
            ),
            hovertemplate='%{text}: %{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>'
        ))

    fig = go.Figure(data=traces)

    # Force a cubic bounding box so the scene renders with equal axes.
    combined_xyz = points_3d
    if show_cameras and camera_translations.size:
        combined_xyz = np.vstack((combined_xyz, camera_translations))

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
        title='Interactive 3D Points from COLMAP Reconstruction',
        width=500,
        height=500
    )
    fig.show()


def filter_points(reconstruction: Reconstruction, filter_outliers: bool = False) -> np.ndarray:
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


def get_camera_translations(reconstruction: Reconstruction) -> np.ndarray:
    # Camera positions need the inverse transform (world_from_cam) to recover centers
    camera_centers = [image.cam_from_world().inverse().translation for image in reconstruction.images.values()]
    return np.stack(camera_centers)

def get_camera_filenames(reconstruction: Reconstruction) -> list[str]:
    # Get camera file names
    camera_filenames = [item.name for item in reconstruction.images.values()]
    return camera_filenames