# Data

This is a log to describe the datasets generated for training and experimentation.

dataset-1.pt
# TODO

...

## dataset-4.pt

This dataset has smaller translation ranges and includes image features.

The dataset was created with the following modifications to `generate_data.py`:
```python
    data = generate_sfm_data(
        model_path=model_path,
        image_dir=image_dir,
        num_samples=1000,
        subset_size=15,
        include_image_features=True,
        translation_range=(-1.0, 1.0),
        random_seed=42,
        output_path=pathlib.Path('data/data-4.pt')
    )
```

then run
```bash
uv run generate_data.py
```

## dataset-5.pt

This dataset has smaller translation ranges and includes image features.

The dataset was created with the following modifications to `generate_data.py`:
```python
    data = generate_sfm_data(
        model_path=model_path,
        image_dir=image_dir,
        num_samples=1000,
        subset_size=15,
        include_image_features=True,
        img_size=100,
        translation_range=(-1.0, 1.0),
        random_seed=42,
        output_path=pathlib.Path('data/data-5.pt')
    )
```

then run
```bash
uv run generate_data.py
```

