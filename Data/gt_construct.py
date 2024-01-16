import cv2
import numpy as np
import pandas as pd
import glob
import os
import argparse

low_x0, low_y0, low_w, low_h, bottom_z, top_z = -500, -500, 1000, 1000, -500, 500

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
        txt_path = ensure_dir(os.path.join(dataset_root, 'SN_cases_0112', f"SN_{timestamp}{SN_num}"))
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

def associate_slices_within_cube(start_z, end_z, image_paths, mask, dataset_root, SN_timestamp, timestamp, i, direction):
    tmp_mask = mask

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
                mask_dir_root = ensure_dir(os.path.join(dataset_root, 'SN_cases_0112', f"SN_{SN_timestamp}{i}", str(timestamp)))
                mask_name = f"{image_path.split('/')[-1].split('.')[-2]}.png"     
                cv2.imwrite(os.path.join(mask_dir_root, mask_name), current_mask * 255)
                no_match = False
                break       # Moving to the next slice
        
        # If can't find any match in this slice, then move on to the next phase
        if no_match:
            break

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

def trace_current_timestamp(mask_candidates, timestamp, image_paths, all_data, dataset_root):
    # filter out all the SN events
    filtered_data = filter_data(all_data[(all_data['time_Myr'] >= timestamp - 1) & (all_data['time_Myr'] <= timestamp)],
                            (low_x0, low_y0, low_w, low_h, bottom_z, top_z))

    for SN_num in range(filtered_data.shape[0]):
        posx_pc = int(filtered_data.iloc[SN_num]["posx_pc"])
        posy_pc = int(filtered_data.iloc[SN_num]["posy_pc"])
        posz_pc = int(filtered_data.iloc[SN_num]["posz_pc"])

        center_slice_z = posz_pc + 500

        anchor_img = read_image_grayscale(image_paths[center_slice_z])
        binary_image = apply_otsus_thresholding(anchor_img)
        num_labels, labels, stats, centroids = find_connected_components(binary_image)

        for i in range(2, num_labels):     
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 1000:
                continue
            
            # x, y = centroids[i]

            x1 = stats[i, cv2.CC_STAT_LEFT] 
            y1 = stats[i, cv2.CC_STAT_TOP] 
            w = stats[i, cv2.CC_STAT_WIDTH] 
            h = stats[i, cv2.CC_STAT_HEIGHT] 

            # tracing this SN in the bubble for the entire cube 
            if SN_center_in_bubble(posx_pc, posy_pc, x1, y1, w, h):
                # it is a new SN case, construct a new profile for the SN case
                mask = labels == i
                if not in_mask_candidates(mask_candidates, mask):
                    mask_candidates.append(mask)

                    # then it should save the mask to the new mask folder
                    mask_dir_root = ensure_dir(os.path.join(dataset_root, 'SN_cases_0112', f"SN_{timestamp}{i}"))
                    mask_name = f"{image_paths[0].split('/')[-1].split('.')[-2]}.png"     
                    cv2.imwrite(os.path.join(mask_dir_root, str(timestamp), mask_name), mask * 255)
                    with open(os.path.join(mask_dir_root, f"SN_{timestamp}{i}_info.txt"), "w") as f:
                        f.write(str(filtered_data.iloc[SN_num]))

                # tracking up       #TODO: see if this can tap, and combine with the SN_center_in_bubble block
                associate_slices_within_cube(center_slice_z - 1, 0, image_paths, mask, dataset_root, timestamp, timestamp, i, -1)
                # tracking down
                associate_slices_within_cube(center_slice_z + 1, 1000, image_paths, mask, dataset_root, timestamp, timestamp, i, 1)
                
    return mask_candidates
            

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


def associate_subsequent_timestamp(timestamp, start_Myr, end_Myr, dataset_root):
    # loop through all slices in the mask folder
    img_prefix = "sn34_smd132_bx5_pe300_hdf5_plt_cnt_0"
    
    SN_ids = retrieve_id(glob.glob(os.path.join(dataset_root, 'SN_cases_0112', f'SN_{timestamp}*'))) 
    
    for SN_id in SN_ids:
        center_z = read_center_z(os.path.join(dataset_root, 'SN_cases_0112', f"SN_{timestamp}{SN_id}", f"SN_{timestamp}{SN_id}_info.txt"), default_z = 0) 
        
        # ignore cases if there's no info file
        if center_z == 0:
            continue
        
        mask = read_image_grayscale(os.path.join(dataset_root, 'SN_cases_0112', f"SN_{timestamp}{SN_id}", str(timestamp), f"{img_prefix}{timestamp}_z{center_z}.png"))

        for time_Myr in range(start_Myr, end_Myr):
            image_paths = sort_image_paths(glob.glob(os.path.join(dataset_root, 'raw_img', str(time_Myr), f"{img_prefix}{time_Myr}_z*.png"))) 

            # tracking up
            associate_slices_within_cube(center_z - 1, 0, image_paths, mask, dataset_root, timestamp, time_Myr, SN_id, -1)
            # tracking down
            associate_slices_within_cube(center_z + 1, 1000, image_paths, mask, dataset_root, timestamp, time_Myr, SN_id, 1)
    


# convert seconds to Megayears
def seconds_to_megayears(seconds):
    return seconds / (1e6 * 365 * 24 * 3600)

# Convert pixel value to pc
def pixel2pc(pixel):
    return (pixel * 10) / 8

def cm2pc(cm):
    return cm * 3.24077929e-19


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



def main(args):
    timestamps = range(int(args.start_Myr), int(args.end_Myr) - 3)  # List of timestamps
    mask_candidates = []

    # Read and process the .dat file logs
    all_data_df = read_dat_log(args.dat_file_root, args.dataset_root)

    for timestamp in timestamps:
        image_paths = sort_image_paths(glob.glob(os.path.join(args.dataset_root, 'raw_img', str(timestamp), '*.png'))) # List of image paths for this timestamp


        # DEBUG 
        print(f"\n\nStart tracing through time {timestamp}")


        mask_candidates = trace_current_timestamp(mask_candidates, timestamp, image_paths, all_data_df, args.dataset_root)

        # DEBUG 
        print(f"Done tracing through time {timestamp}")
        print("Start associating with the subsequent timestamp")

        # associate with all later timestamp
        associate_subsequent_timestamp(timestamp, timestamp + 1, int(args.end_Myr), args.dataset_root)
        
        # DEBUG
        print(f"Done associating for timestamp {timestamp}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_Myr", help="The starting timestamp for the dataset", type = int)             # 200
    parser.add_argument("--end_Myr", help="The ending timestamp for the dataset", type = int)               # 219
    parser.add_argument("--dataset_root", help="The root directory to the dataset")          # "../Dataset"
    parser.add_argument("--dat_file_root", help="The root directory to the SNfeedback files")         # "SNfeedback"

    # python Data/gt_construct.py --start_Myr 200 --end_Myr 219 --dataset_root "../Dataset" --dat_file_root "SNfeedback" > output.txt 2>&1 &
    
    args = parser.parse_args()
    main(args)