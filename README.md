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

## Run Experiments

Experiments set 1: Translation only perturbations

```bash
# Experiments set 1: Translation only perturbations
uv run train.py --config-path conf/e1 --config-name e1_0_base
uv run train.py --config-path conf/e1 --config-name e1_1_gat
uv run train.py --config-path conf/e1 --config-name e1_2_gat_emb256

# Experiments set 2: Rotation only perturbations
uv run train.py --config-path conf/e2 --config-name e2_0_base
uv run train.py --config-path conf/e2 --config-name e2_0_a_base
uv run train.py --config-path conf/e2 --config-name e2_0_b_base
uv run train.py --config-path conf/e2 --config-name e2_1_no_frob_reg
uv run train.py --config-path conf/e2 --config-name e2_1_a_no_frob_reg
uv run train.py --config-path conf/e2 --config-name e2_2_quaternion
uv run train.py --config-path conf/e2 --config-name e2_2_a_quaternion
uv run train.py --config-path conf/e2 --config-name e2_3_quaternion_naive
uv run train.py --config-path conf/e2 --config-name e2_3_a_quaternion_naive

# Experiments set 3: Rotation and Translation perturbations
uv run train.py --config-path conf/e3 --config-name e3_0-NEW
uv run train.py --config-path conf/e3 --config-name e3_1-NEW

# Experiments set 4: With image features
uv run train.py --config-path conf/e4 --config-name e4_0_base
uv run train.py --config-path conf/e4 --config-name e4_1_quat_mse
```

View results with tensorboard:
```bash
tensorboard --logdir runs
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