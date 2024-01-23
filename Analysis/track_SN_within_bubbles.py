import cv2 
import glob
import os
import argparse
import numpy as np
from Data.utils import *

existence_thres = 0.4

class Tracklet:
    def __init__(self, name, time, center_x, center_y, center_z, mask):
        self.name = name
        self.time = time
        self.center = (center_x, center_y, center_z)
        self.mask = mask
        self.volume = {}
        self.explosions = []

    def add_explosion(self, name, time, center_x, center_y, center_z, mask, volume):
        self.explosions.append({
            'time': time,
            'center_position': (center_x, center_y, center_z),
            'mask': mask,
            'volume': volume,
            'track': Tracklet(name, time, center_x, center_y, center_z, mask)
        })
    def add_volume(self, time, volume):
        self.volume[time] = volume

def calculate_iou(mask1, mask2):
    # Implement IOU calculation here
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def load_mask(folder_path, slice_number):
    # Load the mask for the given slice number
    mask_path = os.path.join(folder_path, f'mask_{slice_number}.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask

def sort_SN_img_paths(image_paths):
    slice_image_paths = {}
    for path in image_paths:
        time = int(path.split("/")[-1].split(".")[-2].split("z")[-1])       # the time here actually refers to the z_slice, but it's only a temporary parameter, so didn't change
        slice_image_paths[time] = path
    
    image_paths_sorted = []
    for key in sorted(slice_image_paths):
        image_paths_sorted.append(slice_image_paths[key])
    return image_paths_sorted

def track_existed(parent, tracklets):
    # read the info file
    center_z = pc2pixel(read_info(glob.glob(os.path.join(parent, f"*.txt"))[0], info_col="posz_pc"), x_y_z = "z")
    timestamp = time_Myr2timestamp(read_info(glob.glob(os.path.join(parent, f"*.txt"))[0], info_col="time_Myr")) 

    # read the mask
    current_mask = load_mask(parent, timestamp, f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{timestamp}_z{center_z}.png")

    # compute iou with previous tracks
    for prev_tracklet in tracklets:
        parent_mask = prev_tracklet.mask
        if compute_iou(current_mask, parent_mask) > existence_thres:
            return True
    return False

def process_tracklets(start_timestamp, end_timestamp, interval, dataset_root):
    tracklets = []

    for timestamp in range(start_timestamp, end_timestamp, interval):
        # get all cases folder path for this timestamp
        parent_folders = glob.glob(os.path.join(dataset_root, f"SN_{timestamp}*"))
        
        for parent in parent_folders:
            if not track_existed(parent, tracklets):
                # add new tracklet
                pass
            # add new explosion
            # record info, name, time, xyz, mask
            name = parent.split("/")[-1]
            time = int(name[3:6])
            txt_file = glob.glob(os.path.join(parent, str(time), "*.txt"))[0]
            center_x, center_y, center_z = read_info(txt_file, info_col="posx_pc"), read_info(txt_file, info_col="posy_pc"), read_info(txt_file, info_col="posz_pc")
            
            # TODO: complete mask reading part
            mask = load_mask

            # loop through masks, calc volume
            img_paths = sort_image_paths(glob.glob(os.path.join(parent, str(timestamp), f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0*")))
        # image_paths = sort_image_paths(glob.glob(os.path.join(dataset_root, f'SN_{timestamp}*', str(timestamp), '*.png')))




        timestamp_folder = f'SN_{timestamp}'
        text_file_path = os.path.join(timestamp_folder, 'info.txt')

        if os.path.exists(text_file_path):
            with open(text_file_path, 'r') as f:
                start_time, center_x, center_y, center_z = map(int, f.readline().split())

            center_mask = load_mask(timestamp_folder, center_z)

            current_tracklet = Tracklet(timestamp_folder)
            current_tracklet.add_explosion(start_time, (center_x, center_y, center_z), center_mask)

            for prev_tracklet in tracklets:
                prev_explosion = prev_tracklet.explosions[-1]
                prev_mask = prev_explosion['mask']

                iou = calculate_iou(center_mask, prev_mask)

                if iou > 0.6:
                    current_tracklet.explosions.append(prev_explosion)

            tracklets.append(current_tracklet)

    return tracklets



def main(args):
    result_tracklets = process_tracklets(args.start_timestamp, args.end_timestamp, args.interval, args.dataset_root)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_timestamp", help="Specify the starting timestamp", default = 200, type = int)     
    parser.add_argument("--end_timestamp", help="Specify the end timestamp for tracking", default = 330, type = int)
    parser.add_argument("--interval", help="Specify the interval between timestamps", default = 1, type = int)    
    parser.add_argument("--dataset_root", help="Path to dataset root", default = "../../Dataset")   


    
    
    args = parser.parse_args()
    main(args)