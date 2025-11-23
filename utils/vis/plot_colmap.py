import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from pycolmap import Reconstruction

def plot_colmap_points3D(reconstruction: Reconstruction, filter_outliers: bool = False, show_cameras: bool = False):
    # Extract 3D point coordinates
    points_3d = filter_points(reconstruction, filter_outliers)

    # Extract camera positions
    camera_translations = get_image_translations(reconstruction)
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


def plot_image_subsets(reconstruction: Reconstruction, team_a_ids: set[int], team_b_ids: set[int]):
    team_a_images = [image for image in reconstruction.images.values() if image.image_id in team_a_ids]
    team_b_images = [image for image in reconstruction.images.values() if image.image_id in team_b_ids]

    team_a_camera_centers = np.stack([image.projection_center() for image in team_a_images])
    team_b_camera_centers = np.stack([image.projection_center() for image in team_b_images])

    team_a_camera_names = [item.name for item in team_a_images]
    team_b_camera_names = [item.name for item in team_b_images]

    # Plot cameras
    traces = []
    traces.append(go.Scatter3d(
        x=team_a_camera_centers[:, 0],
        y=team_a_camera_centers[:, 1],
        z=team_a_camera_centers[:, 2],
        mode='markers',
        name='Set A Images',
        text=team_a_camera_names,  # list of strings used in hover template
        marker=dict(
            size=5,
            color='red',
            symbol='diamond',
            opacity=1.0
        ),
        hovertemplate='%{text}: %{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>'
    ))
    traces.append(go.Scatter3d(
        x=team_b_camera_centers[:, 0],
        y=team_b_camera_centers[:, 1],
        z=team_b_camera_centers[:, 2],
        mode='markers',
        name='Set B Images',
        text=team_b_camera_names,  # list of strings used in hover template
        marker=dict(
            size=5,
            color='blue',
            symbol='diamond',
            opacity=1.0
        ),
        hovertemplate='%{text}: %{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>'
    ))

    fig = go.Figure(data=traces)

    # Force a cubic bounding box so the scene renders with equal axes.
    # combined_xyz = points_3d
    # if show_cameras and camera_translations.size:
    #     combined_xyz = np.vstack((combined_xyz, camera_translations))

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
        title='Image Subsets',
        width=500,
        height=500
    )
    fig.show()



def _create_coordinate_frame(translation, rotation_matrix, label, axis_length, colors):
    """
    Helper function to create coordinate frame traces for a single pose.

    Args:
        translation: 3D position of the frame origin
        rotation_matrix: 3x3 rotation matrix
        label: Label for this frame
        axis_length: Length of each axis
        colors: List of 3 colors for X, Y, Z axes

    Returns:
        List of plotly traces representing the coordinate frame
    """
    traces = []
    axis_names = ['X', 'Y', 'Z']

    # Create three axes
    for i, (axis_name, color) in enumerate(zip(axis_names, colors)):
        # Create unit vector along this axis
        axis_dir = rotation_matrix[:, i]

        # Scale by axis length
        axis_end = translation + axis_dir * axis_length

        # Create line from origin to end
        trace = go.Scatter3d(
            x=[translation[0], axis_end[0]],
            y=[translation[1], axis_end[1]],
            z=[translation[2], axis_end[2]],
            mode='lines+markers',
            name=f"{label} - {axis_name}",
            line=dict(color=color, width=5),
            marker=dict(size=[5, 3], color=color),
            hovertemplate=f'{label} - {axis_name} axis<br>Start: ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})<br>End: ({axis_end[0]:.3f}, {axis_end[1]:.3f}, {axis_end[2]:.3f})<extra></extra>',
            showlegend=(i == 0)  # Only show one legend entry per pose
        )
        traces.append(trace)

    # Add a marker at the center
    center_trace = go.Scatter3d(
        x=[translation[0]],
        y=[translation[1]],
        z=[translation[2]],
        mode='markers',
        name=label,
        marker=dict(size=8, color='black', symbol='circle'),
        hovertemplate=f'{label}<br>Position: ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})<extra></extra>',
        showlegend=False
    )
    traces.append(center_trace)

    return traces


def plot_rigid3d_poses(poses, labels=None, show_origin=True, axis_length=0.1):
    """
    Plot one or more COLMAP Rigid3D datapoints as coordinate frames in 3D space.

    Args:
        poses: A single Rigid3d object or a list of Rigid3d objects.
               Each Rigid3d should have rotation_xyzw and translation attributes.
        labels: Optional list of labels for each pose (used in hover text).
        show_origin: If True, plots the origin (0,0,0) with identity rotation.
        axis_length: Length of the coordinate frame axes to display.

    Returns:
        A plotly Figure object showing the poses as coordinate frames.
    """
    # Import here to avoid circular dependencies
    from pycolmap import Rigid3d

    # Handle single pose input
    if isinstance(poses, Rigid3d):
        poses = [poses]

    # If labels not provided, create default labels
    if labels is None:
        labels = [f"Pose {i}" for i in range(len(poses))]
    elif isinstance(labels, str):
        labels = [labels]

    traces = []

    # Add origin if requested
    if show_origin:
        # Identity rotation and zero translation
        origin_traces = _create_coordinate_frame(
            translation=np.array([0.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3),
            label="Origin",
            axis_length=axis_length,
            colors=['red', 'green', 'blue']
        )
        traces.extend(origin_traces)

    # Add each pose
    for i, pose in enumerate(poses):
        # Get translation
        translation = np.array(pose.translation)

        # Get rotation matrix from quaternion
        # pycolmap uses xyzw format for quaternions
        quat = np.array(pose.rotation.quat)  # This should give us the quaternion
        rotation_matrix = pose.rotation.matrix()  # Get rotation matrix directly

        # Create coordinate frame for this pose
        pose_traces = _create_coordinate_frame(
            translation=translation,
            rotation_matrix=rotation_matrix,
            label=labels[i],
            axis_length=axis_length,
            colors=['salmon', 'lightgreen', 'lightblue']
        )
        traces.extend(pose_traces)

    # Create figure
    fig = go.Figure(data=traces)

    # Calculate bounds for all poses
    all_positions = []
    if show_origin:
        all_positions.append(np.array([0.0, 0.0, 0.0]))
    for pose in poses:
        all_positions.append(np.array(pose.translation))

    all_positions = np.array(all_positions)

    # Calculate cubic bounding box
    xyz_min = all_positions.min(axis=0) - axis_length
    xyz_max = all_positions.max(axis=0) + axis_length
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
        title='Camera Poses (Rigid3D)',
        width=800,
        height=800,
        showlegend=True
    )

    return fig


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


def get_image_translations(reconstruction: Reconstruction) -> np.ndarray:
    image_centers = [image.projection_center() for image in reconstruction.images.values()]
    return np.stack(image_centers)


def get_camera_filenames(reconstruction: Reconstruction) -> list[str]:
    # Get camera file names
    camera_filenames = [item.name for item in reconstruction.images.values()]
    return camera_filenames


def plot_colmap_points3D_interactive(reconstruction: Reconstruction, filter_outliers: bool = False, show_cameras: bool = False):
    # Extract 3D point coordinates
    points_3d = filter_points(reconstruction, filter_outliers)

    # Extract camera positions
    camera_translations = get_image_translations(reconstruction)
    camera_names = get_camera_filenames(reconstruction)

    traces = [go.Scatter3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        mode='markers',
        name='3D Points',
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
