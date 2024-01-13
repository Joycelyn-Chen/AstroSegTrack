# AstroSegTrack
- This is the project supporting Joycelyn's master thesis
- We're segmenting and tracking super bubble using the magnetohydrodynamic simulation dataset provided by Alex



## Data Preperation 
### `Data/hdf5topng.py`
- Converting astronomy h5df data type to png images
- Input: data root directory, output root path
- Output: All the sliced images to the designated folder, one per timestemp
- The output images are in grayscale each with size (1000, 1000)

### `Data/gt_construct.py`
- Building the ground truth dataset for the Astro segmentation and tracking model

## Model
### Backbone
- I'm thinking adopting the latest SOTA video object segmentaion model - XMem++
- However, it's still heavily based on XMem, so we'll retrain based on the original XMem post 

---
## ToDO
- [ ] and ignore cases with 1 image only
- track up and down, `gt_constuct.py` line 112, within the line 79 for loop
    - gotta record the anchor slice number
- then associate within the same timestamp
    - read the center slice
    - then track up and down

