import pathlib
import pycolmap

# Create output directory
output_path = pathlib.Path("./output")
output_path.mkdir(exist_ok=True)

# Define db path
database_path = output_path / "database.db"

# Define path to input images
image_dir = pathlib.Path("./input/lego")

# Step 1: Extract Features
pycolmap.extract_features(database_path, image_dir, camera_model='OPENCV')

# Step 2: Feature Matching
pycolmap.match_exhaustive(database_path)

# Step 3: Mapper
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
