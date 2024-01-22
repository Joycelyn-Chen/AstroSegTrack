import cv2 
import glob
import os
import argparse
import numpy as np

class Tracklet:
    def __init__(self, name):
        self.name = name
        self.explosions = []

    def add_explosion(self, time, center_position, mask):
        self.explosions.append({
            'time': time,
            'center_position': center_position,
            'mask': mask
        })

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

def process_tracklets(start_timestamp, end_timestamp, interval):
    tracklets = []

    for timestamp in range(end_timestamp, start_timestamp - 1, -interval):
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
    result_tracklets = process_tracklets(args.start_timestamp, args.end_timestamp, args.interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_timestamp", help="Specify the starting timestamp", type = int) 
    parser.add_argument("--end_timestamp", help="Specify the end timestamp for tracking", type = int)
    parser.add_argument("--interval", help="Specify the interval between timestamps", type = int)       


    
    
    args = parser.parse_args()
    main(args)