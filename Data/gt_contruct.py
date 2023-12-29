import cv2
import numpy as np
import pandas as pd
import glob
import os

def read_image_grayscale(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def apply_otsus_thresholding(image):
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def find_connected_components(binary_image):
    return cv2.connectedComponentsWithStats(binary_image)

def SN_in_dataframe(dataframe, timestamp, x, y, z, tol_error = 10):
    cropped_df = dataframe[(dataframe['time_Myr'] >= timestamp) & (dataframe['time_Myr'] <= timestamp)]         # + 1 if returns nothing
    result_df = cropped_df[(cropped_df['posx_pc'] > x - tol_error) & (cropped_df['posx_pc'] < x + tol_error) & (cropped_df['posy_pc'] > y - tol_error) & (cropped_df['posy_pc'] < y + tol_error) & (cropped_df['posz_pc'] > z - tol_error & (cropped_df['posz_pc'] < z + tol_error))]
    
    print(result_df)

    if(len(result_df) > 0):
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

def process_timestamp(timestamp, image_paths, dataframe, dataset_root):
    mask_candidates = []

    # Process first slice
    first_image = read_image_grayscale(image_paths[0])
    binary_image = apply_otsus_thresholding(first_image)
    num_labels, labels, stats, centroids = find_connected_components(binary_image)
    
    for i in range(1, num_labels):     
        x, y = centroids[i]
        z = -400

        # DEBUG
        print("Processing 1st image...")
        print(f"Component {i}: ({pixel2pc(x)}, {pixel2pc(y)})")

        if SN_in_dataframe(dataframe, timestamp, pixel2pc(x), pixel2pc(y), z,  tol_error = 10):     
            # it is a new SN case
            # construct a new profile for the SN case
            mask = labels == i
            mask_candidates.append(mask)

            # then it should save the mask to the mask folder
            mask_dir_root = ensure_dir(os.path.join(dataset_root, f"SN_{i}", str(timestamp)))
            mask_name = f"{image_paths[0].split('/')[-1].split('.')[-2]}.jpg"     # -2 or -3
            cv2.imwrite(os.path.join(mask_dir_root, mask_name), mask * 255)



    # Process subsequent slices
    for image_path in image_paths[1:]:
        image = read_image_grayscale(image_path)
        binary_image = apply_otsus_thresholding(image)
        num_labels, labels, stats, centroids = find_connected_components(binary_image)

        for i in range(1, num_labels):
            current_mask = labels == i
            for j, candidate_mask in enumerate(mask_candidates):
                iou = compute_iou(current_mask, candidate_mask)
                if iou >= 0.8:
                    # Update mask candidate and output current mask
                    mask_candidates[j] = current_mask

                    mask_dir_root = ensure_dir(os.path.join(dataset_root, f"SN_{j}", str(timestamp)))
                    mask_name = f"{image_path.split('/')[-1].split('.')[-2]}.jpg"     # -2 or -3
                    cv2.imwrite(os.path.join(mask_dir_root, mask_name), current_mask * 255)
                    

# convert seconds to Megayears
def seconds_to_megayears(seconds):
    return seconds / (1e6 * 365 * 24 * 3600)

# Convert pixel value to pc
def pixel2pc(pixel):
    return (pixel * 10) / 8

def cm2pc(cm):
    return cm * 3.24077929e-19

# filter the DataFrame
def filter_data(df, range_coord):
    return df[(df['posx_pc'] > range_coord[0]) & (df['posx_pc'] < range_coord[0] + range_coord[2]) & (df['posy_pc'] > range_coord[1]) & (df['posy_pc'] < range_coord[1] + range_coord[3]) & (df['posz_pc'] > range_coord[4] & (df['posz_pc'] < range_coord[5]))]


def read_dat_log(dat_file_root, dataset_root):
    dat_files = glob.glob(os.path.join(dataset_root, dat_file_root, "*.dat"))

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

    # Convert time to Megayears
    all_data['time_Myr'] = seconds_to_megayears(all_data['time'])

    # Convert 'pos' from centimeters to parsecs
    all_data['posx_pc'] = cm2pc(all_data['posx'])
    all_data['posy_pc'] = cm2pc(all_data['posy'])
    all_data['posz_pc'] = cm2pc(all_data['posz'])

    # Sort the DataFrame by time in ascending order
    all_data.sort_values(by='time_Myr', inplace=True)

    return all_data


                    
start_time = 200
end_time = 201
timestamps = range(start_time, end_time)  # List of timestamps

# File paths parameters
# dataset_root = "/Users/joycelynchen/Desktop/UBC/Research/Program/Dataset/200_210/"
dataset_root = "../Dataset"
dat_file_root = "SNfeedback"


# Read and process the .dat file logs
all_data_df = read_dat_log(dat_file_root, dataset_root)


for timestamp in timestamps:
    image_paths = glob.glob(os.path.join(dataset_root, 'raw_img', str(timestamp), '*.jpg')) # List of image paths for this timestamp

    # sort the image paths accoording to their slice number
    slice_image_paths = {}
    for path in image_paths:
        time = int(path.split("/")[-1].split(".")[-2].split("z")[-1])
        slice_image_paths[time] = path
    
    image_paths_sorted = []
    for key in sorted(slice_image_paths):
        image_paths_sorted.append(slice_image_paths[key])

    
    process_timestamp(timestamp, image_paths_sorted, all_data_df, dataset_root)
