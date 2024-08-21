# AstroSegTrack
- This is the project supporting Joycelyn's master thesis
- We're segmenting and tracking super bubble using the magnetohydrodynamic simulation dataset provided by Alex


# Installation

# Data Preperation 
- Use `data/hdf5tojpg.py` to converte astronomy h5df data type to jpg images
- Input: data root directory, output root path
- Output: All the sliced images to the designated folder
- The output images are in grayscale each with size (256, 256)
- example execution:
```
cd data/
python hdf5tojpg.py --hdf5_root "<path to hdf5 root>" --output_root_dir "<path to output root>" --start_timestamp 200 --end_timestamp 210 --offset 1
```



### `Data/gt_construct.py`
- Building the ground truth dataset for the Astro segmentation and tracking model

## Model
### Backbone
- I'm thinking adopting the latest SOTA video object segmentaion model - XMem++
- However, it's still heavily based on XMem, so we'll retrain based on the original XMem post 

---
## ToDO

# Pipeline steps
- confirm the HDF5 location
- study SBfeedback.dat and look for one SB of interest to track (SB_info.ipynb)
- generate the 2D dens slices for the designated duration (data/hdf5tojpg.py)
- (on a GPU) refer to Astro-3DIS for VOS segmentation implementation 
- transfer the masks back to elephant
- run wholeCube_SN_target_k3d.py to plot the target SB tracking on k3d

