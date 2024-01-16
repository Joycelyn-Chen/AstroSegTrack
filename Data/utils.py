import cv2
import numpy as np
import pandas as pd
import glob
import os

low_x0, low_y0, low_w, low_h, bottom_z, top_z = -500, -500, 1000, 1000, -500, 500


def read_image_grayscale(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image
    except:
        return np.zeros((1000, 1000)) 

def apply_otsus_thresholding(image):
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(binary_image)

def find_connected_components(binary_image):
    return cv2.connectedComponentsWithStats(binary_image)

def SN_in_dataframe(dataframe, timestamp, SN_num, dataset_root, date, x, y, tol_error = 10):
    cropped_df = dataframe[(dataframe['time_Myr'] >= timestamp - 1) & (dataframe['time_Myr'] <= timestamp + 1)]        
    result_df = cropped_df[(cropped_df['posx_pc'] > x - tol_error) & (cropped_df['posx_pc'] < x + tol_error) & (cropped_df['posy_pc'] > y - tol_error) & (cropped_df['posy_pc'] < y + tol_error)]       #  & (cropped_df['posz_pc'] > z - tol_error & (cropped_df['posz_pc'] < z + tol_error))
    
    if(len(result_df) > 0):
        # Store the .dat log for each SN case
        txt_path = ensure_dir(os.path.join(dataset_root, f'SN_cases_{date}', f"SN_{timestamp}{SN_num}"))
        result_df.to_csv(os.path.join(txt_path, f'SNfeedback_{timestamp}{SN_num}.txt'), sep='\t', index=False, encoding='utf-8')
        
        # DEBUG
        print(f"Outputting to file {txt_path}")
        
        return True
    return False

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def in_mask_candidates(mask_candidates, mask):
    for mask_candidate in mask_candidates:
        iou = compute_iou(mask, mask_candidate)
        if iou >= 0.6:      # all mask should not even overlap with each other, so 60% overlapped is a very high threshold
            return True
    return False

# filter the DataFrame
def filter_data(df, range_coord):
    return df[(df['posx_pc'] > range_coord[0]) & (df['posx_pc'] < range_coord[0] + range_coord[2]) & (df['posy_pc'] > range_coord[1]) & (df['posy_pc'] < range_coord[1] + range_coord[3]) & (df['posz_pc'] > range_coord[4]) & (df['posz_pc'] < range_coord[5])]


def within_range(min, max, target):
    if min < target and max > target:
        return True
    return False

# see if the SN center is within the bubble bounding box region
def SN_center_in_bubble(posx_pc, posy_pc, x1, y1, w, h):
    posx_pc = posx_pc + 500
    posy_pc = posy_pc + 500

    if within_range(x1, x1 + w, posx_pc) and within_range(y1, y1 + h, posy_pc):
        return True
    return False

def sort_image_paths(image_paths):
    # sort the image paths accoording to their slice number
    slice_image_paths = {}
    for path in image_paths:
        time = int(path.split("/")[-1].split(".")[-2].split("z")[-1])
        slice_image_paths[time] = path
    
    image_paths_sorted = []
    for key in sorted(slice_image_paths):
        image_paths_sorted.append(slice_image_paths[key])
    return image_paths_sorted

# convert seconds to Megayears
def seconds_to_megayears(seconds):
    return seconds / (1e6 * 365 * 24 * 3600)

# Convert pixel value to pc
def pixel2pc(pixel):
    return (pixel * 10) / 8

def cm2pc(cm):
    return cm * 3.24077929e-19

def read_center_z(SN_info_file, default_z):
    try: 
        with open(SN_info_file, "r") as f:
            data = f.readlines()
        for line in data:
            line = line.strip("\n")
            if(line.split()[0] == "posz_pc"):
                return int(float(line.split()[1]))
            
    except FileNotFoundError as e:
        print(f"File {SN_info_file} not found.")
        return default_z
    
    except Exception as e:
        print(f"Error reading {SN_info_file}")
        return default_z
                    

def retrieve_id(image_paths):
    for i, path in enumerate(image_paths):
        image_paths[i] = path.split("/")[-1][6:]
    return image_paths

