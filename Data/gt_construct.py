import cv2
import numpy as np
import pandas as pd
import glob
import os

def read_image_grayscale(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def apply_otsus_thresholding(image):
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(binary_image)

def find_connected_components(binary_image):
    return cv2.connectedComponentsWithStats(binary_image)

def SN_in_dataframe(dataframe, timestamp, SN_num, dataset_root, x, y, tol_error = 10):
    cropped_df = dataframe[(dataframe['time_Myr'] >= timestamp - 1) & (dataframe['time_Myr'] <= timestamp + 1)]        
    result_df = cropped_df[(cropped_df['posx_pc'] > x - tol_error) & (cropped_df['posx_pc'] < x + tol_error) & (cropped_df['posy_pc'] > y - tol_error) & (cropped_df['posy_pc'] < y + tol_error)]       #  & (cropped_df['posz_pc'] > z - tol_error & (cropped_df['posz_pc'] < z + tol_error))
    
    if(len(result_df) > 0):
        # Store the .dat log for each SN case
        txt_path = ensure_dir(os.path.join(dataset_root, f"SN_{timestamp}{SN_num}"))
        result_df.to_csv(os.path.join(txt_path, f'SNfeedback_{timestamp}{SN_num}.txt'), sep='\t', index=False, encoding='utf-8')
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

def trace_current_timestamp(timestamp, image_paths, dataframe, dataset_root):
    mask_candidates = []

    # Process first slice
    first_image = read_image_grayscale(image_paths[0])
    binary_image = apply_otsus_thresholding(first_image)
    num_labels, labels, stats, centroids = find_connected_components(binary_image)
    
    # DEBUG
    print("Processing 1st image...")

    for i in range(2, num_labels):     
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 1000:
            continue
        
        x, y = centroids[i]
        # z = -400

        # DEBUG
        print(f"Component {i}: ({pixel2pc(x - 400)}, {pixel2pc(y - 400)})")

        if SN_in_dataframe(dataframe, timestamp, i, dataset_root, pixel2pc(x - 400), pixel2pc(y - 400),  tol_error = 30):     
            # it is a new SN case
            # construct a new profile for the SN case
            mask = labels == i
            mask_candidates.append(mask)

            # then it should save the mask to the mask folder
            mask_dir_root = ensure_dir(os.path.join(dataset_root, f"SN_{timestamp}{i}", str(timestamp)))
            mask_name = f"{image_paths[0].split('/')[-1].split('.')[-2]}.jpg"     
            cv2.imwrite(os.path.join(mask_dir_root, mask_name), mask * 255)
        

    # Process subsequent slices
    for image_path in image_paths[1:]:
        image = read_image_grayscale(image_path)
        binary_image = apply_otsus_thresholding(image)
        num_labels, labels, stats, centroids = find_connected_components(binary_image)


        # DEBUG
        img_name = image_path.split("/")[-1]
        print(f"Tracing through time {timestamp}, processing image {img_name}")
        
        for i in range(1, num_labels):
            current_mask = labels == i

            for j, candidate_mask in enumerate(mask_candidates):
                iou = compute_iou(current_mask, candidate_mask)
                if iou >= 0.5:
                    # Update mask candidate and output current mask
                    mask_candidates[j] = current_mask

                    mask_dir_root = ensure_dir(os.path.join(dataset_root, f"SN_{timestamp}{j}", str(timestamp)))
                    mask_name = f"{image_path.split('/')[-1].split('.')[-2]}.jpg"     # -2 or -3
                    cv2.imwrite(os.path.join(mask_dir_root, mask_name), current_mask * 255)
            
                    
def associate_subsequent_timestamp(timestamp, start_yr, end_yr, dataset_root):
    # loop through all slices in the mask folder
    img_prefix = "sn34_smd132_bx5_pe300_hdf5_plt_cnt_0"
    image_paths = glob.glob(os.path.join(dataset_root, 'SN_*', str(timestamp), '*.jpg')) # List of image paths for this timestamp
    for image_path in image_paths:
        SN_num = int(image_path.split("/")[-3].split("_")[-1])
        slice_num = image_path.split("/")[-1].split(".")[-2].split("z")[-1]

        
        # read the pivot mask as binary image
        mask_binary = read_image_grayscale(image_path)

        # find cooresponding slice in the raw img folder
        for time in range(start_yr, end_yr):
            next_raw_img_path = os.path.join(dataset_root, 'raw_img', str(time), f"{img_prefix}{time}_z{slice_num}.jpg")

            # DEBUG
            print(f"Asssociating SN_{SN_num} for timestamp {time}")

            # otsu and connected component the new slice
            image = read_image_grayscale(next_raw_img_path)
            binary_image = apply_otsus_thresholding(image)
            num_labels, labels, stats, centroids = find_connected_components(binary_image)


            # find the component with the most similar iou
            for i in range(num_labels):
                current_mask = labels == i
                if compute_iou(current_mask, mask_binary) >= 0.6:
                    # output the mask for this timestamp
                    mask_binary = current_mask
                    mask_dir_root = ensure_dir(os.path.join(dataset_root, f"SN_{SN_num}", str(time)))
                    mask_name = f"{img_prefix}{time}_z{slice_num}.jpg"
                    cv2.imwrite(os.path.join(mask_dir_root, mask_name), current_mask * 255)
                    break


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


# def identify_SN_cases(df, start_Myr, end_Myr):
#     cropped_df = df[(df['time_Myr'] >= start_Myr) & (df['time_Myr'] <= end_Myr + 1)]
#     return cropped_df.groupby(['posx_pc', 'posy_pc', 'posz_pc']).first().reset_index()



def main():
    start_Myr = 200
    end_Myr = 210
    timestamps = range(start_Myr, end_Myr)  # List of timestamps

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

        # DEBUG 
        print(f"\n\nStart tracing through time {timestamp}")


        trace_current_timestamp(timestamp, image_paths_sorted, all_data_df, dataset_root)

        # DEBUG 
        print(f"Done tracing through time {timestamp}")
        print("Start associating with the subsequent timestamp")

        # associate with all later timestamp
        associate_subsequent_timestamp(timestamp, timestamp + 1, end_Myr, dataset_root)



if __name__ == '__main__':
    main()