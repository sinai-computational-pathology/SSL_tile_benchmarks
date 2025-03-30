# Feature Extraction

Code to extract features from tiles and save them to tensors. For each task and foundation model, it generates for each slide a 2D tensor of shape number of tiles by number of features.
Usage:
```python
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

## Tissue Tile Generation
There are many options to extract tissue tiles and any of them could be used here. In our work we use the strategy from [Campanella et al.](https://www.nature.com/articles/s41591-019-0508-1).
This is a fast method that works well for H&E slides. We provide the code in `extract_tissue.py`. To use:
```python
import extract_tissue
import openslide
help(extract_tissue.make_sample_grid)
slide = openslide.OpenSlide(path/to/a/slide)
# Generate coordinates
base_mpp = extract_tissue.slide_base_mpp(slide)
coord_list = extract_tissue.make_sample_grid(slide, patch_size=224, mpp=0.5, mult=4, base_mpp=base_mpp)
# Plot extraction
extract_tissue.plot_extraction(slide, patch_size=224, mpp=0.5, mult=4, base_mpp=base_mpp)
```