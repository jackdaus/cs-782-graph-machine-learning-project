
# pycolmap Exhaustive API Reference (Latest Version)

This document provides an exhaustive reference for the latest version of `pycolmap` (October 2025), including all method signatures and properties. For the full API, see the [official documentation](https://colmap.github.io/pycolmap/pycolmap.html).

---

## Table of Contents
- [Key Concepts](#key-concepts)
- [Main Classes](#main-classes)
- [Main Functions](#main-functions)
- [Example Workflow](#example-workflow)
- [Additional Resources](#additional-resources)

---

## Key Concepts
- **pycolmap** is a Python interface for COLMAP, providing access to 3D reconstruction, feature extraction, matching, and more.
- The API is object-oriented, with classes for cameras, images, points, reconstructions, and more.

---

## Main Classes

### pycolmap.Database
```python
class pycolmap.Database(path: str)
```
**Methods:**
- `open(path: str) -> None`
- `close() -> None`
- `exists_camera(camera_id: int) -> bool`
- `exists_image(image_id: int | name: str) -> bool`
- `exists_pose_prior(image_id: int) -> bool`
- `exists_keypoints(image_id: int) -> bool`
- `exists_descriptors(image_id: int) -> bool`
- `exists_matches(image_id1: int, image_id2: int) -> bool`
- `exists_inlier_matches(image_id1: int, image_id2: int) -> bool`
- `read_rig(rig_id: int) -> pycolmap.Rig`
- `read_rig_with_sensor(sensor_id: pycolmap.sensor_t) -> pycolmap.Rig | None`
- `read_all_rigs() -> list[pycolmap.Rig]`
- `read_camera(camera_id: int) -> pycolmap.Camera`
- `read_all_cameras() -> list[pycolmap.Camera]`
- `read_frame(frame_id: int) -> pycolmap.Frame`
- `read_all_frames() -> list[pycolmap.Frame]`
- `read_image(image_id: int) -> pycolmap.Image`
- `read_image_with_name(name: str) -> pycolmap.Image | None`
- `read_all_images() -> list[pycolmap.Image]`
- `read_pose_prior(image_id: int) -> pycolmap.PosePrior`
- `read_keypoints(image_id: int) -> np.ndarray[float32[m, n]]`
- `read_descriptors(image_id: int) -> np.ndarray[uint8[m, n]]`
- `read_matches(image_id1: int, image_id2: int) -> np.ndarray[uint32[m, 2]]`
- `read_all_matches() -> tuple[list[int], list[np.ndarray[uint32[m, 2]]]]`
- `read_two_view_geometry(image_id1: int, image_id2: int) -> pycolmap.TwoViewGeometry`
- `read_two_view_geometries() -> tuple[list[int], list[pycolmap.TwoViewGeometry]]`
- `read_two_view_geometry_num_inliers() -> tuple[list[int], list[int]]`
- `write_camera(camera: pycolmap.Camera, use_camera_id: bool = False) -> int`
- `write_image(image: pycolmap.Image, use_image_id: bool = False) -> int`
- `write_pose_prior(image_id: int, pose_prior: pycolmap.PosePrior) -> None`
- `write_keypoints(image_id: int, keypoints: np.ndarray[float32[m, n]]) -> None`
- `write_descriptors(image_id: int, image_id: int, descriptors: np.ndarray[uint8[m, n]]) -> None`
- `write_matches(image_id1: int, image_id2: int, matches: np.ndarray[uint32[m, 2]]) -> None`
- `write_two_view_geometry(image_id1: int, image_id2: int, two_view_geometry: pycolmap.TwoViewGeometry) -> None`
- `update_camera(camera: pycolmap.Camera) -> None`
- `update_image(image: pycolmap.Image) -> None`
- `delete_matches(image_id1: int, image_id2: int) -> None`
- `delete_inlier_matches(image_id1: int, image_id2: int) -> None`
- `clear_all_tables() -> None`
- `clear_cameras() -> None`
- `clear_images() -> None`
- `clear_pose_priors() -> None`
- `clear_descriptors() -> None`
- `clear_keypoints() -> None`
- `clear_matches() -> None`
- `clear_two_view_geometries() -> None`
- `merge(database1: pycolmap.Database, database2: pycolmap.Database, merged_database: pycolmap.Database) -> None`

**Properties:**
- `num_cameras`
- `num_images`
- `num_pose_priors`
- `num_keypoints`
- `num_descriptors`
- `num_matches`
- `num_inlier_matches`
- `num_matched_image_pairs`
- `num_verified_image_pairs`

---

### pycolmap.Reconstruction
```python
class pycolmap.Reconstruction(path: str = None)
```
**Methods:**
- `read(path: str) -> None`
- `write(output_dir: str) -> None`
- `read_text(path: str) -> None`
- `read_binary(path: str) -> None`
- `write_text(path: str) -> None`
- `write_binary(path: str) -> None`
- `num_rigs() -> int`
- `num_cameras() -> int`
- `num_frames() -> int`
- `num_reg_frames() -> int`
- `num_images() -> int`
- `num_reg_images() -> int`
- `num_points3D() -> int`
- `rig(rig_id: int) -> pycolmap.Rig`
- `camera(camera_id: int) -> pycolmap.Camera`
- `frame(frame_id: int) -> pycolmap.Frame`
- `image(image_id: int) -> pycolmap.Image`
- `point3D(point3D_id: int) -> pycolmap.Point3D`
- `exists_image(image_id: int) -> bool`
- `exists_point3D(point3D_id: int) -> bool`
- `tear_down() -> None`
- `add_rig(rig: pycolmap.Rig) -> None`
- `add_camera(camera: pycolmap.Camera) -> None`
- `add_frame(frame: pycolmap.Frame) -> None`
- `add_image(image: pycolmap.Image) -> None`
- `add_point3D(point3D: pycolmap.Point3D) -> None`
- `delete_point3D(point3D_id: int) -> None`
- `delete_observation(image_id: int, point2D_idx: int) -> None`
- `register_image(frame_id: int) -> None`
- `deregister_image(frame_id: int) -> None`
- `normalize(fixed_scale: bool = False, extent: float = 10.0, min_percentile: float = 0.1, max_percentile: float = 0.9, use_images: bool = True) -> pycolmap.Sim3d`
- `transform(new_from_old_world: pycolmap.Sim3d) -> None`
- `compute_centroid(min_percentile: float = 0.0, max_percentile: float = 1.0, use_images: bool = False) -> np.ndarray[float64[3, 1]]`
- `compute_bounding_box(min_percentile: float = 0.0, max_percentile: float = 1.0, use_images: bool = False) -> pycolmap.AlignedBox3d`
- `crop(bbox: pycolmap.AlignedBox3d) -> pycolmap.Reconstruction`
- `find_image_with_name(name: str) -> pycolmap.Image`
- `find_common_reg_image_ids(other: pycolmap.Reconstruction) -> list[tuple[int, int]]`
- `update_point_3d_errors() -> None`
- `compute_num_observations() -> int`
- `compute_mean_track_length() -> float`
- `compute_mean_observations_per_reg_image() -> float`
- `compute_mean_reprojection_error() -> float`
- `import_PLY(path: str) -> None`
- `export_PLY(output_path: str) -> None`
- `extract_colors_for_image(image_id: int, path: str) -> bool`
- `extract_colors_for_all_images(path: str) -> None`
- `create_image_dirs(path: str) -> None`
- `summary() -> str`

**Properties:**
- `rigs`, `cameras`, `frames`, `images`, `points3D`

---

### pycolmap.Camera
```python
class pycolmap.Camera
```
**Properties:**
- `model_id`, `width`, `height`, `params`, `prior_focal_length`, `has_bogus_params`, ...

**Methods:**
- `cam_from_img(image_point: np.ndarray[2, 1]) -> Optional[np.ndarray[2, 1]]`
- `img_from_cam(cam_point: np.ndarray[3, 1]) -> Optional[np.ndarray[2, 1]]`
- `rescale(new_width: int, new_height: int) -> None`
- `summary(write_type: bool = False) -> str`
- `todict(recursive: bool = True) -> dict`

---

### pycolmap.Image
```python
class pycolmap.Image
```
**Properties:**
- `image_id`, `camera_id`, `frame_id`, `data_id`, `camera`, `frame`, `name`, `has_pose`, `points2D`, `num_points3D`, ...

**Methods:**
- `cam_from_world() -> pycolmap.Rigid3d`
- `projection_center() -> np.ndarray[3, 1]`
- `viewing_direction() -> np.ndarray[3, 1]`
- `project_point(point3D: np.ndarray[3, 1]) -> np.ndarray[2, 1] | None`
- `has_camera_id() -> bool`
- `has_camera_ptr() -> bool`
- `reset_camera_ptr() -> None`
- `has_frame_id() -> bool`
- `has_frame_ptr() -> bool`
- `reset_frame_ptr() -> None`
- `num_points2D() -> int`
- `get_observation_point2D_idxs() -> list[int]`
- `get_observation_points2D() -> pycolmap.Point2DList`
- `summary(write_type: bool = False) -> str`
- `todict(recursive: bool = True) -> dict`

---

### pycolmap.Point3D
```python
class pycolmap.Point3D
```
**Properties:**
- `xyz`, `color`, `error`, `track`

**Methods:**
- `summary(write_type: bool = False) -> str`
- `todict(recursive: bool = True) -> dict`

---

### pycolmap.Rigid3d, pycolmap.Sim3d, pycolmap.Rotation3d
**See official docs for all overloaded constructors and methods.**

---

## Main Functions

### Feature Extraction and Matching
- `pycolmap.extract_features(database_path: str, image_path: str, image_names: list[str] = [], camera_mode: pycolmap.CameraMode = CameraMode.AUTO, camera_model: str = 'SIMPLE_RADIAL', reader_options: pycolmap.ImageReaderOptions = ImageReaderOptions(), sift_options: pycolmap.SiftExtractionOptions = SiftExtractionOptions(), device: pycolmap.Device = Device.auto) -> None`
- `pycolmap.match_exhaustive(database_path: str, sift_options: pycolmap.SiftMatchingOptions = SiftMatchingOptions(), matching_options: pycolmap.ExhaustiveMatchingOptions = ExhaustiveMatchingOptions(), verification_options: pycolmap.TwoViewGeometryOptions = TwoViewGeometryOptions(), device: pycolmap.Device = Device.auto) -> None`
- `pycolmap.match_spatial`, `match_vocabtree`, `match_sequential`, `verify_matches`, ...

### Reconstruction
- `pycolmap.incremental_mapping(database_path: str, image_path: str, output_path: str, options: pycolmap.IncrementalPipelineOptions = IncrementalPipelineOptions(), input_path: str = '', initial_image_pair_callback: Callable[[], None] = None, next_image_callback: Callable[[], None] = None) -> dict[int, pycolmap.Reconstruction]`
- `pycolmap.triangulate_points(reconstruction: pycolmap.Reconstruction, database_path: str, image_path: str, output_path: str, clear_points: bool = True, options: pycolmap.IncrementalPipelineOptions = IncrementalPipelineOptions(), refine_intrinsics: bool = False) -> pycolmap.Reconstruction`
- `pycolmap.bundle_adjustment(reconstruction: pycolmap.Reconstruction, options: pycolmap.BundleAdjustmentOptions = BundleAdjustmentOptions()) -> None`

### Dense Reconstruction
- `pycolmap.patch_match_stereo(workspace_path: str, workspace_format: str = 'COLMAP', pmvs_option_name: str = 'option-all', options: pycolmap.PatchMatchOptions = PatchMatchOptions(), config_path: str = '') -> None`
- `pycolmap.stereo_fusion(output_path: str, workspace_path: str, workspace_format: str = 'COLMAP', pmvs_option_name: str = 'option-all', input_type: str = 'geometric', options: pycolmap.StereoFusionOptions = StereoFusionOptions()) -> pycolmap.Reconstruction`
- `pycolmap.poisson_meshing(input_path: str, output_path: str, options: pycolmap.PoissonMeshingOptions = PoissonMeshingOptions()) -> None`
- `pycolmap.dense_delaunay_meshing(input_path: str, output_path: str, options: pycolmap.DelaunayMeshingOptions = DelaunayMeshingOptions()) -> None`

### Pose Estimation and Triangulation
- `pycolmap.estimate_absolute_pose(points2D: np.ndarray[m, 2], points3D: np.ndarray[m, 3], camera: pycolmap.Camera, estimation_options: pycolmap.AbsolutePoseEstimationOptions = AbsolutePoseEstimationOptions()) -> dict | None`
- `pycolmap.refine_absolute_pose(cam_from_world: pycolmap.Rigid3d, points2D: np.ndarray[m, 2], points3D: np.ndarray[m, 3], inlier_mask: np.ndarray[bool[m, 1]], camera: pycolmap.Camera, refinement_options: pycolmap.AbsolutePoseRefinementOptions = AbsolutePoseRefinementOptions(), return_covariance: bool = False) -> dict | None`
- `pycolmap.estimate_and_refine_absolute_pose(points2D: np.ndarray[m, 2], points3D: np.ndarray[m, 3], camera: pycolmap.Camera, estimation_options: pycolmap.AbsolutePoseEstimationOptions = AbsolutePoseEstimationOptions(), refinement_options: pycolmap.AbsolutePoseRefinementOptions = AbsolutePoseRefinementOptions(), return_covariance: bool = False) -> dict | None`
- `pycolmap.triangulate_point(cam1_from_world: np.ndarray[3, 4], cam2_from_world: np.ndarray[3, 4], cam_point1: np.ndarray[2, 1], cam_point2: np.ndarray[2, 1]) -> np.ndarray[3, 1] | None`

### Utility
- `pycolmap.set_random_seed(seed: int) -> None`
- `pycolmap.Timer`, `pycolmap.logging`, ...

---

## Example Workflow
```python
import pycolmap
# Feature extraction
pycolmap.extract_features('output/database.db', 'input/01_lego_small')
# Feature matching
pycolmap.match_exhaustive('output/database.db')
# Incremental mapping
pycolmap.incremental_mapping('output/database.db', 'input/01_lego_small', 'output/0')
# Load and inspect reconstruction
rec = pycolmap.Reconstruction('output/0')
print(rec.num_images(), rec.num_points3D())
```

---

## Additional Resources
- [COLMAP Main Page](https://colmap.github.io/index.html)
- [pycolmap API Reference](https://colmap.github.io/pycolmap/pycolmap.html)
- [COLMAP Database Format](https://colmap.github.io/database.html)
- [COLMAP Camera Models](https://colmap.github.io/cameras.html)
- [COLMAP Output Format](https://colmap.github.io/format.html)

---
This file is auto-generated from the official pycolmap documentation for quick reference. For detailed usage, always consult the [official docs](https://colmap.github.io/pycolmap/pycolmap.html).
