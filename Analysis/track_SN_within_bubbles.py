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
        self.explosions = []

    # def add_explosion(self, name, time, center_x, center_y, center_z, mask, volume):
    #     self.explosions.append({
    #         'time': time,
    #         'center_position': (center_x, center_y, center_z),
    #         'mask': mask,
    #         'volume': volume,
    #         'track': Tracklet(name, time, center_x, center_y, center_z, mask)
    #     })
    def add_explosion(self, explosion):
        self.explosions.append(explosion)
    

def track_existed(parent, center_z, timestamp, tracks):
    # read the mask
    current_mask = load_mask(parent, timestamp, f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{timestamp}_z{center_z}.png")

    # compute iou with previous tracks
    for prev_tracklet in tracks:
        parent_mask = prev_tracklet.mask
        if compute_iou(current_mask, parent_mask) > existence_thres:
            return prev_tracklet
    return None



def process_tracklets(start_timestamp, end_timestamp, interval, dataset_root):
    tracks = []

    for timestamp in range(start_timestamp, end_timestamp, interval):
        # get all cases folder path for this timestamp
        parent_folders = glob.glob(os.path.join(dataset_root, f"SN_{timestamp}*"))
        
        for parent in parent_folders:
            # record info, name, time, xyz, mask
            name = parent.split("/")[-1]
            txt_file = glob.glob(os.path.join(parent, "*.txt"))[0]
            center_x, center_y, center_z = pc2pixel(read_info(txt_file, info_col="posx_pc"), x_y_z="x"), pc2pixel(read_info(txt_file, info_col="posy_pc"), x_y_z="y"), pc2pixel(read_info(txt_file, info_col="posz_pc"), x_y_z="z")
            
            current_tracklet = track_existed(parent, center_z, timestamp, tracks)
            if current_tracklet is None:
                mask = load_mask(parent, timestamp, f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{timestamp}_z{center_z}.png")
                
                # add a new tracklet
                current_tracklet = Tracklet(name, timestamp, center_x, center_y, center_z, mask)
                
            # add new explosion
            # loop through masks, incrementing in timestamp, calc volume
            explosion = []
            for time in sorted(list(map(int, list_folders(parent)))):
                img_paths = glob.glob(os.path.join(parent, time, "*.png"))
                volume_pix = volume_sum_in_mask(img_paths, parent, time)
                explosion.append({'time': time, 'center_position': (center_x, center_y, center_z), 'mask': mask, 'volume': volume_pix})
            current_tracklet.add_explosion(explosion) 
            

            # for prev_tracklet in tracks:
            #     prev_explosion = prev_tracklet.explosions[-1]
            #     prev_mask = prev_explosion['mask']

            #     iou = compute_iou(mask, prev_mask)

            #     if iou > 0.6:
            #         current_tracklet.explosions.append(prev_explosion)

            tracks.append(current_tracklet)

    return tracks



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