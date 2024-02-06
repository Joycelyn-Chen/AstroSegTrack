import os
import argparse
import yt
from data.utils import *

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from utils import *

low_x0, low_y0, low_w, low_h, bottom_z, top_z = -500, -500, 1000, 1000, -500, 500



def filter_data(df, time_range, posz_pc_range):
    # Filter the DataFrame based on specified conditions
    return df[(df['time_Myr'].between(time_range[0], time_range[1])) & (df['posz_pc'].between(posz_pc_range[0], posz_pc_range[1]))]


def associate_slices_within_cube(start_z, end_z, image_paths, mask, mask_dir_root, timestamp, direction):
    tmp_mask = mask
    volume = 0

    for img_path_id in range(start_z, end_z, direction):    # direction: +1 tracking down, -1 tracking up
        image_path = image_paths[img_path_id]
        image = read_image_grayscale(image_path)
        binary_image = apply_otsus_thresholding(image)
        num_labels, labels, stats, centroids = find_connected_components(binary_image)

        no_match = True

        for label in range(2, num_labels):
            current_mask = labels == label
            if compute_iou(current_mask, tmp_mask) >= 0.6:      # if found a match in this slice
                tmp_mask = current_mask
                mask_name = f"{image_path.split('/')[-1].split('.')[-2]}.png"     
                cv2.imwrite(os.path.join(mask_dir_root, str(timestamp), mask_name), current_mask * 255)

                # log volume for each slice
                area = stats[label, cv2.CC_STAT_AREA]
                volume += area
                no_match = False
                break       # Moving to the next slice
        
        # If can't find any match in this slice, then move on to the next phase
        if no_match:
            break
    return volume

def trace_current_timestamp(timestamp, image_paths, filtered_data, dataset_root):

    for SN_num in range(filtered_data.shape[0]):        # should execute only once
        posx_pc = int(filtered_data.iloc[SN_num]["posx_pc"])
        posy_pc = int(filtered_data.iloc[SN_num]["posy_pc"])
        posz_pc = int(filtered_data.iloc[SN_num]["posz_pc"])

        center_slice_z = pc2pixel(posz_pc, x_y_z = "z")

        anchor_img = read_image_grayscale(image_paths[center_slice_z])
        binary_image = apply_otsus_thresholding(anchor_img)
        num_labels, labels, stats, centroids = find_connected_components(binary_image)

        for i in range(2, num_labels):     
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 100:
                continue
            
            # x, y = centroids[i]

            x1 = stats[i, cv2.CC_STAT_LEFT] 
            y1 = stats[i, cv2.CC_STAT_TOP] 
            w = stats[i, cv2.CC_STAT_WIDTH] 
            h = stats[i, cv2.CC_STAT_HEIGHT] 

            # tracing this SN in the bubble for the entire cube 
            if SN_center_in_bubble(pc2pixel(posx_pc, x_y_z = "x"), pc2pixel(posy_pc, x_y_z = "y"), x1, y1, w, h):
                # it is a new SN case, construct a new profile for the SN case
                mask = labels == i

                # then it should save the mask to the new mask folder
                mask_dir_root = ensure_dir(os.path.join(dataset_root, 'Isolated_case', f"SN_{timestamp}{i}"))
                mask_name = f"{image_paths[0].split('/')[-1].split('.')[-2]}.png"     
                cv2.imwrite(os.path.join(mask_dir_root, str(timestamp), mask_name), mask * 255)
                with open(os.path.join(mask_dir_root, f"SN_{timestamp}{i}_info.txt"), "w") as f:
                    f.write(str(filtered_data.iloc[SN_num]))

                # tracking up       
                upper_volume = associate_slices_within_cube(center_slice_z - 1, 0, image_paths, mask, mask_dir_root, timestamp, -1)
                # tracking down
                lower_volume = associate_slices_within_cube(center_slice_z, 1000, image_paths, mask, mask_dir_root, timestamp, 1)
                break
                
    return upper_volume + lower_volume + area, mask


def associate_next_timestamp(timestamp, start_timestamp, end_timestamp, incr, dataset_root, date):
    # loop through all slices in the mask folder
    img_prefix = "sn34_smd132_bx5_pe300_hdf5_plt_cnt_0"
    
    SN_ids = retrieve_id(glob.glob(os.path.join(dataset_root, f'SN_cases_{date}', f'SN_{timestamp}*'))) 
    
    for SN_id in SN_ids:  
        center_z = pc2pixel(read_info(os.path.join(dataset_root, f'SN_cases_{date}', f"SN_{timestamp}{SN_id}", f"SN_{timestamp}{SN_id}_info.txt"), info_col = "posz_pc"), x_y_z = "z")
        # ignore cases if there's no info file
        if center_z == top_z:
            continue
        
        mask_dir_root = os.path.join(dataset_root, 'Isolated_case', f"SN_{timestamp}{SN_id}")
        mask = read_image_grayscale(os.path.join(mask_dir_root, str(timestamp), f"{img_prefix}{timestamp}_z{center_z}.png"))


        for time in range(start_timestamp, end_timestamp, incr):
            # Debug
            print(f"Associating case SN_{timestamp}{SN_id}: {time}")


            image_paths = sort_image_paths(glob.glob(os.path.join(dataset_root, 'raw_img', str(time), f"{img_prefix}{time}_z*.png"))) 

            # tracking up
            associate_slices_within_cube(center_z - 1, 0, image_paths, mask, mask_dir_root, timestamp, -1)
            # tracking down
            associate_slices_within_cube(center_z, 1000, image_paths, mask, mask_dir_root, timestamp, 1)
    return volume



def segment_and_accumulate_areas(start_timestamp, df, dataset_root, timestamp_bound):
    accumulated_areas = {}
    timestamps = range(start_timestamp, start_timestamp + timestamp_bound + 1)  # Adjust end_timestamp as needed
    blob_disappeared = False

    for timestamp in timestamps:
        if blob_disappeared:
            break

        hdf5_file = f"{dataset_root}/file_{timestamp}.hdf5"  # Adjust path as needed
        blob_position = df[['posx_pc', 'posy_pc', 'posz_pc']].iloc[0]  # Assuming single row in df

        # Assuming trace_current_timestamp() processes the hdf5 file, segments the blob, and returns the area and a mask
        
        image_paths = sort_image_paths(glob.glob(os.path.join(args.dataset_root, 'raw_img', str(timestamp), '*.jpg'))) # List of image paths for this timestamp

        volume, center_mask = trace_current_timestamp()

        if timestamp == start_timestamp:
            previous_mask = center_mask
        else:
            iou = compute_iou(previous_mask, center_mask)  # Assuming a function to compute Intersection over Union
            if iou < 0.2:
                blob_disappeared = True
                continue
            previous_mask = center_mask

        accumulated_areas[timestamp] = accumulated_areas.get(timestamp, 0) + volume

        # Save mask to 'isolated_masks' folder
        mask_filename = f"{dataset_root}/isolated_masks/mask_{timestamp}.png"
        cv2.imwrite(mask_filename, center_mask.astype(np.uint8) * 255)  # Convert boolean mask to uint8

    return accumulated_areas, start_timestamp, timestamp - 1  # Return the range of timestamps where the blob was present

def plot_accumulated_volumes(accumulated_areas):
    import matplotlib.pyplot as plt

    times = list(accumulated_areas.keys())
    volumes = list(accumulated_areas.values())

    plt.plot(times, volumes)
    plt.xlabel('Time (Myr)')
    plt.ylabel('Accumulated Volume')
    plt.title('Accumulated Volume Over Time')
    plt.show()

def count_data_records(df, start_timestamp, end_timestamp, posx, posy, posz):
    # Filter DataFrame based on time and position criteria
    filtered_df = df[(df['time_Myr'].between(start_timestamp, end_timestamp)) & 
                     (df['posx_pc'] == posx) & (df['posy_pc'] == posy) & (df['posz_pc'] == posz)]
    return len(filtered_df)

def main(args):
    all_data_df = read_dat_log(args.dat_file_root, args.dataset_root)
    
    start_time_Myr = timestamp2time_Myr(args.start_timestamp)
    
    filtered_df = filter_data(all_data_df, time_range = (start_time_Myr - (args.interval / 10), start_time_Myr), posz_pc_range = (10 * (int(args.center_z_pc / 10) + 1), 10 * (int(args.center_z_pc / 10) - 1)))

    if not filtered_df.empty:
        accumulated_areas, start_ts, end_ts = segment_and_accumulate_areas(args.start_timestamp, filtered_df, args.dataset_root, args.timestamp_bound)
        plot_accumulated_volumes(accumulated_areas)

        # Assuming posx_pc, posy_pc, posz_pc are the positions of the blob in the filtered_df
        posx, posy, posz = filtered_df[['posx_pc', 'posy_pc', 'posz_pc']].iloc[0]
        data_count = count_data_records(all_data_df, start_ts, end_ts, posx, posy, posz)
        print(f"Count of data records between timestamps {start_ts} and {end_ts}: {data_count}")
    else:
        print("No data records match the specified criteria.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", help="The root directory to the dataset")          # "../Dataset"
    parser.add_argument("--dat_file_root", help="The root directory to the SNfeedback files, relative to dataset root")         # "SNfeedback"
    parser.add_argument("--start_timestamp", help="Specify the starting timestamp", default = 204, type = int)     
    parser.add_argument("--timestamp_bound", help="Specify the end timestamp for tracking", default = 60, type = int)
    parser.add_argument("--interval", help="Specify the interval between timestamps", default = 1, type = int)
    parser.add_argument("--center_z_pc", help="Specify the center position of SN in pc", type = int)            # -44

    parser.add_argument("--delta_t", help="Specify the delta t", default = 0.01, type = float) 
    parser.add_argument("--track_Myr", help="Specify how many Myr you wanna track", default = 6, type = int)    
    parser.add_argument("--dataset_root", help="Path to dataset root", default = "../../Dataset")   
    parser.add_argument("--dat_file_root", help="The root directory to the SNfeedback files, relative to dataset root")         # "SNfeedback"
    parser.add_argument("--output_root", help="Path to output root", default = "../../Dataset/ProjectionPlots")   
    parser.add_argument("--hdf5_root", help="Path to HDF5 root", default = "/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc")   
    
    # python observe_SN_blast.py --start_timestamp 201 --end_timestamp 210 --interval 1 --delta_t 0.1 --track_Myr 6 --dataset_root "../../Dataset" --dat_file_root "SNfeedback" --output_root "../../Dataset/Explosions" --hdf5_root /home/joy0921/Desktop/Dataset/200_360/finer_time_200_360_original
    args = parser.parse_args()
    main(args)
