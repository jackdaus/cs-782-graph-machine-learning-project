# AI Coding Agent Instructions

This project explores 3D reconstruction using the `pycolmap` library. The primary workflow happens within Jupyter Notebooks.

## Key Concepts

- **Goal**: To experiment with different `pycolmap` functionalities for Structure from Motion (SfM) and Multi-View Stereo (MVS).
- **Core Library**: `pycolmap`. The agent should be familiar with its API or refer to the official documentation for detailed API usage: [https://colmap.github.io/pycolmap/index.html](https://colmap.github.io/pycolmap/index.html)
- **Environment**: The project is primarily developed in Jupyter Notebooks (`.ipynb` files). Key libraries include `pycolmap`, `numpy`, `matplotlib`, and `opencv-python`.

## Developer Tooling
Python packages are managed with uv. If a new package is needed, use `uv add ...` in the command line to add the package to this project.

## Project Structure

- `input/`: Contains input image datasets, organized in subdirectories (e.g., `input/lego`).
- `output/`: All generated files, including the COLMAP database (`database.db`) and reconstruction models, are stored here.
- `*.ipynb`: Jupyter notebooks for experimentation. `scratch.ipynb` is for general testing, and other notebooks like `01_basic_pipeline.ipynb` contain more structured workflows.

## Developer Workflow

The typical workflow follows the COLMAP pipeline and is executed in notebook cells:

1.  **Setup**: Define input and output paths using `pathlib`. The main output is `output/database.db`.
    ```python
    import pathlib
    output_path = pathlib.Path("./output")
    database_path = output_path / "database.db"
    image_dir = pathlib.Path("./input/lego")
    ```

2.  **Feature Extraction**: Extract features from images using `pycolmap.extract_features`.
    ```python
    pycolmap.extract_features(database_path, image_dir)
    ```

3.  **Feature Matching**: Match features between images. `match_exhaustive` is commonly used.
    ```python
    pycolmap.match_exhaustive(database_path)
    ```

4.  **Database Inspection**: Use `pycolmap.Database` to inspect the results of the previous steps, such as reading keypoints, descriptors, and matches.

5.  **Reconstruction**: Perform sparse reconstruction (Structure from Motion) using `pycolmap.incremental_mapping` (this step is not in all notebooks but is a key part of the pipeline).

6.  **Visualization**: Use `matplotlib` and `opencv-python` (`cv2`) to visualize images, keypoints, and matches.

When assisting, focus on adding or modifying cells within the existing notebooks to explore different aspects of the `pycolmap` library.
