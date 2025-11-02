import pathlib
import pycolmap
import time

# Paths
image_dir = pathlib.Path("./data/01_nerf/02_lego_large")

# Create output directory
run_id = time.time_ns() // 1_000_000
output_path = pathlib.Path(f"./output/01_nerf/02_lego_large/{run_id}")
output_path.mkdir(exist_ok=True, parents=True)

# Define db path
database_path = output_path / "database.db"

# Step 1: Extract Features
pycolmap.extract_features(database_path, image_dir, camera_model='OPENCV', camera_mode=pycolmap.CameraMode.SINGLE)

# Step 2: Feature Matching
pycolmap.match_exhaustive(database_path)

# Step 3: Mapper
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
