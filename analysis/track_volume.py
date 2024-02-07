import os
import argparse
import yt
import glob
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

def trace_first_timestamp(timestamp, image_paths, filtered_data, output_root):

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
                mask_dir_root = ensure_dir(os.path.join(output_root, f"SN_{timestamp}{i}"))
                mask_name = f"{image_paths[0].split('/')[-1].split('.')[-2]}.png"     
                cv2.imwrite(os.path.join(mask_dir_root, str(timestamp), mask_name), mask * 255)
                with open(os.path.join(mask_dir_root, f"SN_{timestamp}{i}_info.txt"), "w") as f:
                    f.write(str(filtered_data.iloc[SN_num]))

                # tracking up       
                upper_volume = associate_slices_within_cube(center_slice_z - 1, 0, image_paths, mask, mask_dir_root, timestamp, -1)
                # tracking down
                lower_volume = associate_slices_within_cube(center_slice_z, 1000, image_paths, mask, mask_dir_root, timestamp, 1)
                break
                
    return upper_volume + lower_volume, mask, (x1, y1, w, h)


def associate_next_timestamp(case_timestamp, timestamp, dataset_root, output_root):
    # loop through all slices in the mask folder
    img_prefix = "sn34_smd132_bx5_pe300_hdf5_plt_cnt_0"
    
    SN_ids = retrieve_id(glob.glob(os.path.join(output_root, f'SN_{case_timestamp}*'))) 
    
    for SN_id in SN_ids:  
        center_z = pc2pixel(read_info(os.path.join(output_root, f"SN_{case_timestamp}{SN_id}", f"SN_{case_timestamp}{SN_id}_info.txt"), info_col = "posz_pc"), x_y_z = "z")
        # ignore cases if there's no info file
        if center_z == top_z:
            continue
        
        mask_dir_root = os.path.join(output_root, f"SN_{case_timestamp}{SN_id}")
        mask = read_image_grayscale(os.path.join(mask_dir_root, str(case_timestamp), f"{img_prefix}{case_timestamp}_z{center_z}.png"))

        image_paths = sort_image_paths(glob.glob(os.path.join(dataset_root, 'raw_img', str(timestamp), f"{img_prefix}{timestamp}_z*.png"))) 
        upper_volume = associate_slices_within_cube(center_z - 1, 0, image_paths, mask, mask_dir_root, timestamp, -1)
        lower_volume = associate_slices_within_cube(center_z, 1000, image_paths, mask, mask_dir_root, timestamp, 1)

            
    return upper_volume + lower_volume



def segment_and_accumulate_areas(start_timestamp, filtered_df, dataset_root, timestamp_bound, output_root, disappear_thres):
    accumulated_areas = {}
    timestamps = range(start_timestamp + 1, start_timestamp + timestamp_bound + 1)  # Adjust end_timestamp as needed
    blob_disappeared = False

    image_paths = sort_image_paths(glob.glob(os.path.join(args.dataset_root, 'raw_img', str(start_timestamp), '*.jpg'))) # List of image paths for this timestamp
    volume, center_mask, bbox = trace_first_timestamp(start_timestamp, image_paths, filtered_df, output_root)
    previous_mask = center_mask

    #DEBUG
    print(f"Done tracing first timestamp {start_timestamp}...")

    for timestamp in timestamps:
        if blob_disappeared:
            break
        
        volume, center_mask = associate_next_timestamp(start_timestamp, timestamp, dataset_root)
        if compute_iou(previous_mask, center_mask) < disappear_thres:
            blob_disappeared = True
            continue
        previous_mask = center_mask

        accumulated_areas[timestamp] = accumulated_areas.get(timestamp, 0) + volume         #TODO: might need to change this

        #DEBUG
        print(f"Done tracing {timestamp}... volume = {volume}")


    return accumulated_areas, start_timestamp, timestamp - 1, bbox  # Return the range of timestamps where the blob was present

def plot_accumulated_volumes(accumulated_areas, output_root):
    times = list(accumulated_areas.keys())
    volumes = list(accumulated_areas.values())

    plt.plot(times, volumes, 'bo')
    plt.xlabel('Time (Myr)')
    plt.ylabel('Accumulated Volume (pixels)')
    plt.title('Accumulated Volume Over Time')
    # plt.show()
    plt.savefig(os.path.join(output_root, 'volume.png'))

    #DEBUG
    print(f"Volume chart saved at: {os.path.join(output_root, 'volume.png')}")

def count_data_records(df, start_time_Myr, end_time_Myr, bbox):
    # Filter DataFrame based on time and position criteria
    filtered_df = df[(df['time_Myr'].between(start_time_Myr, end_time_Myr))] # & df['posz_pc'].between() 
    
    for i in range(len(filtered_df)):
        posx_pc = int(filtered_df.iloc[i]['posx_pc'])
        posy_pc = int(filtered_df.iloc[i]['posy_pc'])
        count = 0
        if SN_center_in_bubble(pc2pixel(posx_pc, x_y_z = "x"), pc2pixel(posy_pc, x_y_z = "y"), bbox[0], bbox[1], bbox[2], bbox[3]):
            count += 1
    
    return count

def main(args):
    all_data_df = read_dat_log(args.dat_file_root, args.dataset_root)
    
    start_time_Myr = timestamp2time_Myr(args.start_timestamp)
    print(f"z range: {10 * (int(args.center_z_pc / 10) + 1)} ~ {10 * (int(args.center_z_pc / 10) - 1)}")
    
    filtered_df = filter_data(all_data_df, time_range = (start_time_Myr - (args.interval / 10), start_time_Myr), posz_pc_range = (10 * (int(args.center_z_pc / 10) + 1), 10 * (int(args.center_z_pc / 10) - 1)))

    if not filtered_df.empty:
        _ = ensure_dir(args.output_root)
        accumulated_volumes, start_ts, end_ts, bbox = segment_and_accumulate_areas(args.start_timestamp, filtered_df, args.dataset_root, args.timestamp_bound, args.output_root, args.disappear_thres)
        plot_accumulated_volumes(accumulated_volumes, args.output_root)

        # Assuming posx_pc, posy_pc, posz_pc are the positions of the blob in the filtered_df
        
        data_count = count_data_records(all_data_df, start_ts, end_ts, bbox)
        print(f"\nCount of data records between timestamps {start_ts} and {end_ts}: {data_count}")
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
    parser.add_argument("--output_root", help="Path to output root", default = "../../Dataset/Isolated_case")
    parser.add_argument("--disappear_thres", help="Specify disappear iou threshold", default = 0.2, type = float) 


  
    # python analysis/track_volume.py --dataset_root "../../Dataset" --dat_file_root "SNfeedback" --output_root "../../Dataset/Isolated_case" --start_timestamp 204 --timestamp_bound 60 --interval 1 --center_z_pc -44
    args = parser.parse_args()
    main(args)
