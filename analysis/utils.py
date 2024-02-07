import os
import argparse
import yt
from data.utils import *

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def read_dat_log(dat_file_root, dataset_root):
    # Only 1 log file is enough, cause they're actually copies of each other
    # dat_files = glob.glob(os.path.join(dataset_root, dat_file_root, "*.dat"))
    dat_files = [os.path.join(dataset_root, dat_file_root, "SNfeedback.dat")]
    
    # Initialize an empty DataFrame
    all_data = pd.DataFrame()

    # Read and concatenate data from all .dat files
    for dat_file in dat_files:
        # Assuming space-separated values in the .dat files
        df = pd.read_csv(dat_file, delim_whitespace=True, header=None,
                        names=['n_SN', 'type', 'n_timestep', 'n_tracer', 'time',
                                'posx', 'posy', 'posz', 'radius', 'mass'])
        
        # Convert the columns to numerical
        df = df.iloc[1:]
        df['n_SN'] = df['n_SN'].map(int)
        df['type'] = df['type'].map(int)
        df['n_timestep'] = df['n_timestep'].map(int)
        df['n_tracer'] = df['n_tracer'].map(int)
        df['time'] = pd.to_numeric(df['time'],errors='coerce')
        df['posx'] = pd.to_numeric(df['posx'],errors='coerce')
        df['posy'] = pd.to_numeric(df['posy'],errors='coerce')
        df['posz'] = pd.to_numeric(df['posz'],errors='coerce')
        df['radius'] = pd.to_numeric(df['radius'],errors='coerce')
        df['mass'] = pd.to_numeric(df['mass'],errors='coerce')
        all_data = pd.concat([all_data, df], ignore_index=True)
        all_data = all_data.drop(df[df['n_tracer'] != 0].index)

    # Convert time to Megayears
    all_data['time_Myr'] = seconds_to_megayears(all_data['time'])

    # Convert 'pos' from centimeters to parsecs
    all_data['posx_pc'] = cm2pc(all_data['posx'])
    all_data['posy_pc'] = cm2pc(all_data['posy'])
    all_data['posz_pc'] = cm2pc(all_data['posz'])

    # Sort the DataFrame by time in ascending order
    all_data.sort_values(by='time_Myr', inplace=True)

    return all_data

def timestamp2time_Myr(timestamp):
    return (timestamp - 200) * 0.1 + 191

def time_Myr2timestamp(time_Myr):
    return round(10 * (time_Myr - 191) + 200)

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

def retrieve_id(image_paths):
    for i, path in enumerate(image_paths):
        image_paths[i] = path.split("/")[-1][6:]
    return image_paths