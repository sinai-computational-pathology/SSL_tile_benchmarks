# Feature Extraction

Code to extract features from tiles and save them to tensors. For each task and foundation model, it generates for each slide a 2D tensor of shape number of tiles by number of features.
Usage:
```
python make_tensors.py \
       --slide_data path/to/slide/data/file/for/this/task.csv \
       --tile_data path/to/tile/data/file/for/this/task.csv \
       --encoder a_foundation_model \
       --bsize 1024
       --workers 10
```

## `slide_data`
Task specific slide level data. Must contain the following columns:
- `slide_path`: full path to slide
- `slide`: unique slide identifier
- `tensor_root`: full path to root for that data. Need to add the encoder type
- `tensor_name`: name of tensor file without path

## `tile_data`
Task specific tile level data. Must contain the following columns:
- `slide`: unique slide identifier, same as in the `slide_data` file
- `x`: x coordinate
- `y`: y coordinate
- `level`: pyramid level at which to extract data
- `mult`: factor for tile resize
