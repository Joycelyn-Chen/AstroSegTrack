# import pandas as pd
import os
# import matplotlib.pyplot as plt
import yt
import numpy as np
import os 
# import matplotlib.colors as colors
import k3d
import numpy as np
from k3d import matplotlib_color_maps
import cv2 as cv

from utils import *


# Initialization
# elephant
hdf5_root = "/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc"

start_timestamp = 206
end_timestamp = 207

for timestamp in range(start_timestamp, end_timestamp, 1):
    # timestamp = 211
    time_Myr = timestamp2Myr(timestamp) 

    DEBUG = True
    # Inputting the raw HDF5 file

    if(DEBUG):
        print("Reading raw HDF5 input...")

    ds = yt.load(os.path.join(hdf5_root, 'sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{}'.format(timestamp)))

    center = [0, 0, 0] * yt.units.pc
    arb_center = ds.arr(center, 'code_length')
    xlim = 256
    ylim = 256
    zlim= 256
    left_edge = arb_center + ds.quan(-500, 'pc')
    right_edge = arb_center + ds.quan(500, 'pc')
    obj = ds.arbitrary_grid(left_edge, right_edge, dims=(256,256,256))

    if(DEBUG):
        print("getting velz and dens, waiting for spec...")

    # retrieve the center (256, 256, 256) grid
    x_range_scaled = (0, 256) 
    y_range_scaled = (0, 256)  
    z_range_scaled = (0, 256)

    center_x, center_y, center_z = 128, 128, 128

    velz_cube, dens_cube, temp_cube = get_velz_dens(obj, x_range_scaled, y_range_scaled, z_range_scaled)
    # new_velx, new_vely = get_velx_vely(obj, x_range_scaled, y_range_scaled, z_range_scaled)

    # Reading all the SN within the last Myr
    filename = "SNfeedback.dat"

    all_data = read_SNfeedback(hdf5_root=hdf5_root, filename=filename)

    low_x0, low_y0, low_w, low_h, bottom_z, top_z = 0, 0, 1000, 1000, 0, 1000
    range_coord = [low_x0, low_y0, low_w, low_h, bottom_z, top_z]

    start_Myr = time_Myr - 1
    end_yr = start_Myr + 1

    if(DEBUG):
        print("filtering SNs...")

    # Filter data based on specified conditions
    filtered_data = all_data[(all_data['time_Myr'] >= start_Myr) & (all_data['time_Myr'] <= end_yr)]
    # filtered_data = filter_data(all_data[(all_data['time_Myr'] >= start_Myr) & (all_data['time_Myr'] <= end_yr)],
                                # (low_x0, low_y0, low_w, low_h, bottom_z, top_z))

    if(DEBUG):
        print("here's the SN cases")
        print(filtered_data)

    converted_points = list(zip(
        pc2pix_256(filtered_data['posx_pc']) + 128,
        pc2pix_256(filtered_data['posy_pc']) + 128,
        pc2pix_256(filtered_data['posz_pc']) + 128
    ))

    if(DEBUG):
        print(converted_points)
        print("now generating masks for whole cube...")

    # process all bubbles in the entire cube
    # read density slice
    lower_b = 0
    upper_b = 256
    cube_size = 256
    input_point = (center_x, center_y)
    mask_cube = np.zeros((cube_size, cube_size, upper_b - lower_b))
    dilated_stack = np.zeros((cube_size, cube_size, upper_b - lower_b))

    for current_z in range(upper_b - lower_b):
        dens_slice = normalize4thresholding(dens_cube[:, :, current_z + lower_b]) 

        if(DEBUG):
            cv2.imwrite(f"tmp/dens_{current_z}.jpg", dens_slice)
        # dens_img = cv2.imread(f"tmp/dens_{current_z}.png")
        
        
        # threshold + connected component
        binary_mask = apply_otsus_thresholding(dens_slice)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        # retrieve all mask only
        i = 0
        x, y, w, h, area = stats[i]
        binary_mask = labels == i
        binary_mask = ~binary_mask
        mask_cube[:, :, current_z] = binary_mask
        # binary_mask = binary_mask * 255



    # cast the ring of mask over velz and dens array, plot out the velocity profile, with integrated density as a function of velz  
    velz_cube_roi = np.where(mask_cube, velz_cube[:, :, lower_b:upper_b], np.nan)
    dens_cube_roi = np.where(mask_cube, dens_cube[:, :, lower_b:upper_b], np.nan)

    if(DEBUG):
        print("Done")
        print("reading the target blob masks...")

    # Read all the masks slices into a 3D mask array


    # lower_b = 0
    # upper_b = 255
    # cube_size = 256
    mask_target = np.zeros((cube_size, cube_size, upper_b - lower_b))

    # TODO: If there's coord mismatch could be from here
    # TODO: read all slices from each timestamp from the saved output masks, stored them in "mask"
    # each saved mask update the corresponding mask z slice

    # read all the png files within the mask folder for current timestamp
    # mask_root = f"/home/joy0921/Desktop/Dataset/VOS_output/astro_0219/SN_20915_{timestamp}"
    mask_root = f"/home/joy0921/Desktop/Dataset/img_pix256/masks/{timestamp}"
    mask_files = [file for file in os.walk(mask_root)][0][2]

    for mask_file in mask_files:
        if mask_file.endswith(".png"):  # only consider png files
            mask_slice = cv.imread(os.path.join(mask_root, mask_file), cv.IMREAD_GRAYSCALE)
            
            # resize the mask_slice if it's not in size (256, 256)
            if mask_slice.shape!= (cube_size, cube_size):
                mask_slice = cv.resize(mask_slice[:, :], (cube_size, cube_size))
            
            # check if the mask_slice is a binary array or not, convert to binary if not
            if mask_slice.max() > 1:
                mask_slice = mask_slice / 255
                # _, mask_slice = cv.threshold(mask_slice, 0, 1, cv.THRESH_BINARY)
            
            # z_coord = int(pc2pix_256(int(mask_file.split(".")[0].split("_")[-1][1:])))
            z_coord = int(mask_file.split('.')[0])
            
            # then store to mask array
            mask_target[:, :, z_coord] = mask_slice


    if(DEBUG):
        print("Done")
        print("Now generating the k3d plot for all...")

    # velz_roi = np.where(mask, velz_cube[:, :, lower_b:upper_b], np.nan)
    dens_cube_roi = np.where(mask_cube, dens_cube[:, :, lower_b:upper_b], np.nan)
    dens_target_roi = np.where(mask_target, dens_cube[:, :, lower_b:upper_b], np.nan)

    # Visualize in 3D using k3d

    whole_cube_coords = np.argwhere(~np.isnan(dens_cube_roi))
    target_coords = np.argwhere(~np.isnan(dens_target_roi))

    values_cube = np.log10(dens_cube_roi[whole_cube_coords[:, 0], whole_cube_coords[:, 1], whole_cube_coords[:, 2]])
    values_target = np.log10(dens_target_roi[target_coords[:, 0], target_coords[:, 1], target_coords[:, 2]])

    cube_points = k3d.points(positions=whole_cube_coords,
                            point_size=0.5,
                            shader='3d',
                            opacity=0.2,
                            color_map=matplotlib_color_maps.Viridis,
                            attribute=values_cube,
                            ) # color=0x3f6bc5

    target_points = k3d.points(positions=target_coords,
                            point_size=0.8,
                            shader='3d',
                            opacity=1.0,
                            color_map=matplotlib_color_maps.Viridis,
                            attribute=values_target,
                            ) # color=0x3f6bc5

    SB_center = k3d.points(positions = converted_points, 
                            point_size=3.0,
                            shader='3d',
                            opacity=1.0,
                            color=0xc30010)     # original point: [149, 178, 141]

    plot = k3d.plot(grid=(0, 0, 0, 10, 10, 10),
                    axes=['X', 'Y', 'Z'])


    plot += cube_points
    plot += target_points
    plot += SB_center

    # plot.display()

    with open(f'k3d_html/{time_Myr}.html','w') as fp:
        fp.write(plot.get_snapshot())

    if(DEBUG):
        print("Done. Plot file stored at {}".format(f'k3d_html/{time_Myr}.html'))

        