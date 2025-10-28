import pathlib
import enlighten
import pycolmap
import rerun as rr
import cv2
import numpy as np
from pycolmap import logging

# Set up rerun. Spawn the Rerun viewer window.
rr.init("rerun_example_my_data", spawn=True)

# Create output directory
output_path = pathlib.Path("./output")
output_path.mkdir(exist_ok=True)

# Define db path
database_path = output_path / "database.db"

# Define path to input images
image_dir = pathlib.Path("./input/lego")

# Log all images to Rerun
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
image_paths = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

# Ensure all images and keypoints are logged at the same time on the timeline!
# This is not an iterative process, so we can just log everthing as happening in the same logical moment.
rr.set_time("stable_time", duration=0) 

# for img_path in image_paths:
#     img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
#     if img is None:
#         print(f"Warning: failed to read {img_path}")
#         continue
#     # Convert BGR -> RGB for correct colors
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # Use the filename stem as the entity path to keep things organized
#     rr.log(f"input/{img_path.stem}/image", rr.Image(img_rgb).compress(jpeg_quality=75))

pycolmap.set_random_seed(0)

# Step 1: Extract Features
pycolmap.extract_features(database_path, image_dir, camera_model="OPENCV")

# Log extracted features (keypoints) per image to Rerun
try:
    db = pycolmap.Database(str(database_path))
    images = db.read_all_images()
    # Build mapping from image name -> image_id
    name_to_id = {img.name: img.image_id for img in images}

    feat_logged = 0
    for img_path in image_paths:
        name = img_path.name
        img_id = name_to_id.get(name)

        # Log the image
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: failed to read {img_path}")
        # Convert BGR -> RGB for correct colors
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Use the image id to keep entity organized in Rerun
        rr.log(f"input/{img_id}/image", rr.Image(img_rgb).compress(jpeg_quality=75))

        if img_id is None:
            # This can happen if the database is stale or filenames differ
            print(f"Warning: no DB entry for {name}")
            continue
        kpts = db.read_keypoints(img_id)
        if getattr(kpts, 'size', 0) == 0:
            continue
        # Take the first two columns as pixel coordinates (x, y)
        xy = kpts[:, :2].astype("float32", copy=False)
        rr.log(f"input/{img_id}/keypoints", rr.Points2D(xy, radii=1))
        feat_logged += 1
    print(f"Logged keypoints for {feat_logged} images to Rerun")
except Exception as e:
    # Keep the pipeline robust even if feature logging fails
    print(f"Feature logging to Rerun failed: {e}")

# Step 2: Feature Matching
pycolmap.match_exhaustive(database_path)

# Log matcher results (pairwise correspondences) to Rerun as composite match panels
try:
    with pycolmap.Database(str(database_path)) as db:
        images = db.read_all_images()
        id_to_name = {img.image_id: img.name for img in images}
        name_to_id = {img.name: img.image_id for img in images}

        # Helper functions from COLMAP docs
        def image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
            if image_id1 > image_id2:
                return 2147483647 * image_id2 + image_id1
            else:
                return 2147483647 * image_id1 + image_id2

        # Collect candidate pairs to visualize. We try consecutive images by name order
        # and fall back to scanning a subset of all pairs.
        pairs_to_try = []
        ordered_names = [p.name for p in image_paths]
        for a, b in zip(ordered_names[:-1], ordered_names[1:]):
            if a in name_to_id and b in name_to_id:
                pairs_to_try.append((name_to_id[a], name_to_id[b]))

        # If very few images or consecutive pairs fail, add a few more pairs.
        if len(pairs_to_try) < 5:
            ids = [name_to_id[n] for n in ordered_names if n in name_to_id]
            for i in range(len(ids)):
                for j in range(i + 1, min(i + 4, len(ids))):  # small banded window
                    pairs_to_try.append((ids[i], ids[j]))

        # Deduplicate while preserving order
        seen = set()
        unique_pairs = []
        for p in pairs_to_try:
            if p not in seen:
                unique_pairs.append(p)
                seen.add(p)

        logged_pairs = 0
        # Limit how many panels to log to avoid spamming the viewer
        MAX_PANELS = 8
        MAX_MATCHES_PER_PANEL = 300

        for id1, id2 in unique_pairs:
            if logged_pairs >= MAX_PANELS:
                break
            pair_id = image_ids_to_pair_id(id1, id2)
            try:
                # Prefer verified inliers if available via two-view geometries
                matches = None
                tvg = db.read_two_view_geometry(id1, id2)
                if tvg is not None and hasattr(tvg, "inlier_matches") and tvg.inlier_matches is not None:
                    matches = tvg.inlier_matches
                else:
                    # Fall back to raw descriptor matches if present
                    matches = db.read_matches(pair_id)

                if matches is None or len(matches) == 0:
                    continue

                # Fetch keypoints
                kpts1 = db.read_keypoints(id1)
                kpts2 = db.read_keypoints(id2)
                if getattr(kpts1, "size", 0) == 0 or getattr(kpts2, "size", 0) == 0:
                    continue

                # Read and prepare images (RGB)
                name1 = id_to_name[id1]
                name2 = id_to_name[id2]
                img1_bgr = cv2.imread(str(image_dir / name1), cv2.IMREAD_COLOR)
                img2_bgr = cv2.imread(str(image_dir / name2), cv2.IMREAD_COLOR)
                if img1_bgr is None or img2_bgr is None:
                    continue
                img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
                img2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                H = max(h1, h2)
                W = w1 + w2
                canvas = np.zeros((H, W, 3), dtype=np.uint8)
                canvas[:h1, :w1] = img1
                canvas[:h2, w1:w1 + w2] = img2

                # Prepare matched point coordinates
                m = matches
                if isinstance(m, list):
                    m = np.array(m)
                if m.ndim != 2 or m.shape[1] != 2:
                    # Unexpected format; skip safely
                    continue

                if len(m) > MAX_MATCHES_PER_PANEL:
                    idx = np.linspace(0, len(m) - 1, MAX_MATCHES_PER_PANEL, dtype=int)
                    m = m[idx]

                pts1 = kpts1[m[:, 0], :2]
                pts2 = kpts2[m[:, 1], :2]
                pts2_offset = pts2.copy()
                pts2_offset[:, 0] += w1  # shift x by width of left image

                # Draw correspondences on the RGB canvas with a rainbow color gradient
                # Color is determined by the Y coordinate of the LEFT image keypoint (top->bottom maps across the rainbow)

                def _hsv_to_rgb_uint8(h: float, s: float = 1.0, v: float = 1.0) -> tuple:
                    """Convert HSV (0-1 ranges) to an (R,G,B) uint8 tuple."""
                    # Wrap hue just in case
                    h = float(h) % 1.0
                    s = float(s)
                    v = float(v)
                    i = int(h * 6.0)
                    f = (h * 6.0) - i
                    i = i % 6
                    p = v * (1.0 - s)
                    q = v * (1.0 - f * s)
                    t = v * (1.0 - (1.0 - f) * s)
                    if i == 0:
                        r, g, b = v, t, p
                    elif i == 1:
                        r, g, b = q, v, p
                    elif i == 2:
                        r, g, b = p, v, t
                    elif i == 3:
                        r, g, b = p, q, v
                    elif i == 4:
                        r, g, b = t, p, v
                    else:
                        r, g, b = v, p, q
                    return (int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0)))

                # Normalize Y on the left image to [0,1]
                y_vals = pts1[:, 1]
                y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
                y_span = max(1e-6, y_max - y_min)

                for (x1, y1), (x2, y2) in zip(pts1, pts2_offset):
                    p1 = (int(round(x1)), int(round(y1)))
                    p2 = (int(round(x2)), int(round(y2)))
                    # Map Y (top->bottom) to hue (0->1) to sweep the full rainbow
                    y_norm = (float(y1) - y_min) / y_span
                    color_rgb = _hsv_to_rgb_uint8(y_norm, 1.0, 1.0)
                    cv2.line(canvas, p1, p2, color=color_rgb, thickness=1, lineType=cv2.LINE_AA)
                # Optionally draw small circles at keypoints
                for (x1, y1) in pts1:
                    cv2.circle(canvas, (int(round(x1)), int(round(y1))), 2, color=(0, 255, 0), thickness=-1)
                for (x2, y2) in pts2_offset:
                    cv2.circle(canvas, (int(round(x2)), int(round(y2))), 2, color=(255, 0, 0), thickness=-1)

                pair_label = f"{pathlib.Path(name1).stem}_vs_{pathlib.Path(name2).stem}"
                rr.log(f"matches/{pair_label}", rr.Image(canvas).compress(jpeg_quality=75))
                logged_pairs += 1
            except Exception as pair_err:
                # Skip problematic pairs without breaking the whole pipeline
                print(f"Warning: failed to visualize matches for pair ({id1}, {id2}): {pair_err}")
                continue

        if logged_pairs == 0:
            print("No match panels were logged to Rerun (no matches found or readable).")
        else:
            print(f"Logged {logged_pairs} match panels to Rerun under 'matches/'.")
except Exception as e:
    print(f"Match logging to Rerun failed: {e}")

def incremental_mapping_with_pbar(database_path, image_path, sfm_path):
    with pycolmap.Database(str(database_path)) as database:
        num_images = database.num_images

    # Simple timeline counter for Rerun so we can scrub the SfM progression
    step = 0

    def _bump_time():
        nonlocal step
        rr.set_time("sfm_step", sequence=step)
        step += 1

    # Where to write/read intermediate snapshots of the reconstruction
    snapshot_root = pathlib.Path(sfm_path) / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)

    def _latest_snapshot_dir() -> pathlib.Path | None:
        try:
            subdirs = [p for p in snapshot_root.iterdir() if p.is_dir()]
            if not subdirs:
                return None
            # Prefer numerically named subfolders; otherwise fall back to mtime
            numeric = []
            other = []
            for p in subdirs:
                try:
                    numeric.append((int(p.name), p))
                except ValueError:
                    other.append(p)
            if numeric:
                return sorted(numeric, key=lambda x: x[0])[-1][1]
            return max(other, key=lambda p: p.stat().st_mtime)
        except Exception:
            return None

    def _log_all_cameras_from_model_dir(model_dir: pathlib.Path) -> None:
        try:
            rec = pycolmap.Reconstruction(str(model_dir))
        except Exception:
            return
        # Each call advances the timeline
        _bump_time()
        # Log each registered image pose as a Transform3D under camera/{image_name}
        for img_id, img in rec.images.items():
            try:
                name = getattr(img, "name", f"image_{img_id}")
                T = img.cam_from_world()
                t = [float(x) for x in T.translation]
                q_xyzw = [float(x) for x in T.rotation.quat]  # x,y,z,w order
                rr.log(
                    f"input/{img_id}",
                    rr.Transform3D(
                        translation=t,
                        rotation=rr.Quaternion(xyzw=q_xyzw),
                        relation=rr.TransformRelation.ChildFromParent,
                    ),
                )
                # Log the camera intrinsics
                camera = rec.cameras(img.camera_id)
                if camera is None:
                    print(f"Warning: no camera found for image with id {img_id}")
                rr.log(
                    f"input/{img_id}/image",
                    rr.Pinhole(
                        resolution=[camera.width, camera.height],
                        focal_length=camera.params[:2],
                        principal_point=camera.params[2:],
                    ),
                )
            except Exception as e:
                # Skip any problematic image without failing the whole callback
                print(f"[{__file__}:318] {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # We use enlighten to log progress in the command line
    with enlighten.Manager() as manager:
        with manager.counter(
            total=num_images, desc="Images registered:"
        ) as pbar:
            pbar.update(0, force=True)
            # Configure pipeline to write snapshots at each registration step
            options = pycolmap.IncrementalPipelineOptions()
            options.snapshot_path = str(snapshot_root)
            options.snapshot_frames_freq = 1  # write a snapshot after every registered image

            def on_initial_image_pair():
                # Try to log cameras from the first snapshot (if any)
                snap = _latest_snapshot_dir()
                if snap is not None:
                    _log_all_cameras_from_model_dir(snap)
                # Keep CLI progress in sync (initial pair registers 2 images)
                pbar.update(2)

            def on_next_image():
                # Log cameras from the latest snapshot so the viewer updates incrementally
                snap = _latest_snapshot_dir()
                if snap is not None:
                    _log_all_cameras_from_model_dir(snap)
                pbar.update(1)

            reconstructions = pycolmap.incremental_mapping(
                database_path,
                image_path,
                sfm_path,
                options=options,
                initial_image_pair_callback=on_initial_image_pair,
                next_image_callback=on_next_image,
            )
    return reconstructions

# # Step 3: Mapper
# maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
recs = incremental_mapping_with_pbar(database_path, image_dir, output_path)
# alternatively, use:
# import custom_incremental_pipeline
# recs = custom_incremental_pipeline.main(
#     database_path, image_path, sfm_path
# )
for idx, rec in recs.items():
    logging.info(f"#{idx} {rec.summary()}")

# print(maps)
