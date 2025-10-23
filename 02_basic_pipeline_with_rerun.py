import pathlib
import pycolmap
import rerun as rr
import cv2

# Set up rerun
rr.init("rerun_example_my_data", spawn=True)
# TODO the rerun logging! see https://rerun.io/examples/3d-reconstruction/structure_from_motion

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

for img_path in image_paths:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: failed to read {img_path}")
        continue
    # Convert BGR -> RGB for correct colors
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Use the filename stem as the entity path to keep things organized
    rr.log(f"input/images/{img_path.stem}", rr.Image(img_rgb))

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
        if img_id is None:
            # This can happen if the database is stale or filenames differ
            print(f"Warning: no DB entry for {name}")
            continue
        kpts = db.read_keypoints(img_id)
        if getattr(kpts, 'size', 0) == 0:
            continue
        # Take the first two columns as pixel coordinates (x, y)
        xy = kpts[:, :2].astype("float32", copy=False)
        rr.log(f"input/images/{img_path.stem}/keypoints", rr.Points2D(xy, radii=1))
        feat_logged += 1
    print(f"Logged keypoints for {feat_logged} images to Rerun")
except Exception as e:
    # Keep the pipeline robust even if feature logging fails
    print(f"Feature logging to Rerun failed: {e}")

# # Step 2: Feature Matching
# pycolmap.match_exhaustive(database_path)

# # Step 3: Mapper
# maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)

# print(maps)
