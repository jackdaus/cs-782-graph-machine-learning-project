import pathlib
import pycolmap
import rerun as rr

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

# Step 1: Extract Features
pycolmap.extract_features(database_path, image_dir, camera_model="OPENCV")

# Step 2: Feature Matching
pycolmap.match_exhaustive(database_path)

# Step 3: Mapper
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)

print(maps)
