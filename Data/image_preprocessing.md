# Image Preprocessing

## Raw Data
- Location: `elephant.ws.ok.ubc.ca:/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc`
    - HDF5 file: `sn34_smd132_bx5_pe300_hdf5_plt_cnt_0200` (0200 - 0990)
    - log file: `SNfeedback.*.dat` (8 files in total)

## Convert HDF5 to PNG
- use `hdf5tojpg.py` to convert HDF5 simulation dataset to 2D images.
- Execute: `python Data/hdf5topng.py --input_dir "/home/joy0921/Desktop/Dataset/200_360/finer_time_200_360_original" --output_root_dir "/home/joy0921/Desktop/Dataset/200_360/200_360_png" --file_prefix "sn34_smd132_bx5_pe300_hdf5_plt_cnt_" --start_Myr 200 --end_Myr 210 --offset 1 --xlim 1000 --ylim 1000 --zlim 1000 --extension ".jpg" > output.txt 2>&1 &`
- default output as JPG files, replace with `--extension ".png"` to output as PNG files

## Thresholding + Connected Component
- use `Data/gt_construct.py` to perform thresholding cutoff and link the same group together as a label with connected component with opencv, and construct the ground truth dataset for the video object segmentation model.
    - Execute: `python Data/gt_construct.py --start_Myr 200 --end_Myr 219 --dataset_root "../Dataset" --dat_file_root "SNfeedback" --date 0116 > output.txt 2>&1 &`
- `Data/gt_construct_bk.py` is the previous working version for constructing the ground truth dataset, when the input was RGB image. Haven't ignore the case where there's only 1 mask per case, and the pixel and pc units wasn't aligned now

## Pinpointing the explosion position on column density image
- use `Analysis/label_explosions.py` to label the explosion cases for the previous Myr on the column density map
    - Execute: `python Analysis/label_explosions.py --begin_timestamp 200 --end_timestamp 360 --incr 10 --hdf5_root "/home/joy0921/Desktop/Dataset/200_360/finer_time_200_360_original" --output_dir "/home/joy0921/Desktop/Dataset/200_360/ProjectionPlots" --dataset_root "../../Dataset" --dat_file_root "SNfeedback" > output.txt 2>&1 &` 

## Helper code
### `png2jpg.py`
- I discover after generating the dataset that in the static image dataset, they put the raw images and the mask images under the same directory, the pair has the same image name, but with different extention (eg. img: a.jpg, mask: a.png)
- Execute `python Data/png2jpg.py --path_to_folder "../Dataset/raw_img"`

### `count_img.py`
- use this program to count the total amount of image within the folder
- Execute `python Data/count_img.py --path_to_folder path --duplicates True`

### `remove_jpg.py`
- use this program to remove the JPG files within the dataset that doesn't have a corresponding mask associated
- Execute: `python Data/remove_jpg.py --path "../Dataset/astro"`
