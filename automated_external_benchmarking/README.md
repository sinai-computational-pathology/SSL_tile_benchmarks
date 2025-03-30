# Automated External Benchmarking

## Container

Docker or Singularity containers are acceptable. Our code will run on the local cluster using Singularity. The official [nvidia-pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) or [MONAI](https://hub.docker.com/r/projectmonai/monai/tags) containers are a good place to start.

## `inference.py` Script

This script should run in the container provided by the user.
The script accepts three main arguments:
- `slide_data` path to input csv file which lists the slides to encode. It contains four columns: `slide` a unique slide id which will also be used to name the tensors and matches with the `tile_data` csv files; `slide_path` the path to the slide file; `mult` a rescaling factor necessary if the right magnification is not available in the slide; `level` the level to extract pixel data from the slide.
- `tile_data` path to input csv file with tile information for all tiles listed in the `slide_data` file.
- `output`: path to output directory where .pth files will be saved.
For each slide in the `slide_data` csv, a `.pth` binary tensor file will be generated in the `output` directory.
We provide an [example script](https://github.com/fuchs-lab-public/OPAL/blob/main/SSL_benchmarks/automated_external_benchmarking/inference.py) which the users can modify to their needs.

## Testing

Below we provide minimal examples of tile and slide csv files based on public data from openslide.
Download the slide:
```bash
wget https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs
```

### Slide Data
```text
slide,slide_path,mult,level
CMU-1.svs,CMU-1.svs,1.,0
```

### Tile Data
```text
slide,x,y
CMU-1.svs,0,0
CMU-1.svs,224,224
```

### Run
```bash
python inference.py --slide_data slide_data.csv --tile_data tile_data.csv
```
