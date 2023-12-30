# Image Preprocessing

## Raw Data
- Location: `elephant.ws.ok.ubc.ca:/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc`
    - HDF5 file: `sn34_smd132_bx5_pe300_hdf5_plt_cnt_0200` (0200 - 0990)
    - log file: `SNfeedback.*.dat` (8 files in total)

## Convert HDF5 to PNG
- use `Data/hdf5topng.py --input_dir input_path --output_dir output_path`

## Thresholding + Connected Component
- use `Data/gt_construct.py` to perform thresholding cutoff and link the same group together as a label with connected component with opencv, and construct the ground truth dataset for the video object segmentation model.
