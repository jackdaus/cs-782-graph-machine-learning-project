# About

todo

## Generating Data

To generate the data from colmap reconstructions, use the `generate_data.py` script.

(#TODO describe downloading colmap data)

To generate the 3 datasets used in the experiments.
```bash
# Dataset with only translations perturbations applied
uv run generate_data.py --config-name v1_trans_only

# Dataset with only rotations perturbations applied
uv run generate_data.py --config-name v2_rot_only

# Dataset with both rotations and translations perturbations applied
uv run generate_data.py --config-name v3_rot_and_trans
```

## Data Generation Details

Configuration is in `conf/data_gen/base.yaml`. Key parameters:
- `paths.model` - COLMAP reconstruction directory
- `paths.images` - Source images directory  
- `paths.output` - Output directory (default: `data/generated`)
- `generation.num_samples` - Number of training samples
- `generation.subset_size` - Images per graph
- `features.img_size` - Image resize dimension

Output files:
- `image_features.pt` - Shared image feature cache
- `samples.pt` - Training samples

--- 

# OLD

--- 

This repo explores usage of [pycolmap](https://colmap.github.io/pycolmap/index.html). 

- Here is a [basic usage example](https://github.com/colmap/colmap/blob/f8edccaa36909713b9d3930e1ca65cb364a38b26/python/examples/example.py) from the colmap authors.
- Here are some useful [docs from DeepWiki](https://deepwiki.com/colmap/colmap/5-python-interface).

# Quick Start

You can run the basic script that will run the default colmap SfM pipeline using the nerf lego dataset.
To run the experimental setup,
```bash
uv run .\01_basic_pipeline.py
```

## Generating Training Data

The notebooks use pre-generated training data. To regenerate the data with different parameters:

```bash
uv run python generate_data_legacy.py
```

This script:
1. Loads a COLMAP reconstruction
2. Creates random subsets of images
3. Applies random rotations and translations to create synthetic training pairs
4. Saves the data to `data/data.pt`

You can modify the parameters in `generate_data.py` to change:
- Number of samples
- Subset size
- Translation range
- Random seed
- Whether to include image features

## Saving PyG datasets

`GraphPairDataset` instances can be serialized once and reloaded without rebuilding:

```python
dataset.save("data/data.pt")
# Later, on a trusted file
dataset = GraphPairDataset.load("data/data.pt", map_location="cpu")
```

PyTorch 2.6+ defaults `torch.load` to `weights_only=True`, which triggers an `UnpicklingError` for PyG classes such as `torch_geometric.data.data.DataEdgeAttr`. Our loader automatically allowlists those globals and, if needed, falls back to `weights_only=False` with a warning—only do this for checkpoints you trust to avoid arbitrary code execution.
