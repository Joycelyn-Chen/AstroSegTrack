import os
import argparse
import yt
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import json
from utils import *
import torch
from segment_anything import sam_model_registry, SamPredictor


low_x0, low_y0, low_w, low_h, bottom_z, top_z = -500, -500, 1000, 1000, -500, 500


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = '/home/joy0921/Desktop/segment-anything/checkpoints/sam_vit_h_4b8939.pth'
DEBUG = True

# Initialize the model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

def sam_and_save_mask(image_path, output_path, input_box, input_point):
    input_box = np.array(input_box)

    # Read and preprocess the image
    image_array = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    predictor.set_image(image_array)

    # Define the input point and label
    input_point = np.array([input_point])
    input_label = np.array([1])

    # Point input
    # Predict the mask
    # masks, scores, logits = predictor.predict(
    #     point_coords=input_point,
    #     point_labels=input_label,
    #     multimask_output=True,
    # )

    # bbox input
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )

    best_mask = masks[0]
    best_mask = (best_mask * 255).astype(np.uint8)

    # Save the mask as an image
    cv2.imwrite(output_path, best_mask)

    #DEBUG
    if(DEBUG):
        plt.figure(figsize=(10, 10))
        plt.imshow(image_array)
        show_mask(best_mask, plt.gca())
        show_box(input_box, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.axis('off')
        plt.savefig("tmp.png")
        plt.clf()

def plot_accumulated_volumes(accumulated_areas, output_root):
    times = list(accumulated_areas.keys())
    volumes = list(accumulated_areas.values())

    plt.plot(times, volumes, 'bo-')
    plt.xlabel('Time (Myr)')
    plt.ylabel('Accumulated Volume (pixels)')
    plt.title('Accumulated Volume Over Time')
    # plt.show()
    plt.savefig(os.path.join(output_root, 'volume.png'))

    #DEBUG
    print(f"Volume chart saved at: {os.path.join(output_root, 'volume.png')}")



def main(args):
    
    start_timestamp = time_Myr2timestamp(args.start_time_Myr)
    end_timestamp = time_Myr2timestamp(args.end_time_Myr) + 1

    for timestamp in (start_timestamp, end_timestamp. args.interval):
        #obj = read_hdf5(args.hdf5_root, args.file_prefix, timestamp)
        ds = yt.load(os.path.join(args.hdf5_root, '{}{}'.format(args.file_prefix, timestamp)))

        center = [0, 0, 0] * yt.units.pc
        arb_center = ds.arr(center, 'code_length')
        xlim = 256
        ylim = 256
        zlim= 256
        left_edge = arb_center + ds.quan(-500, 'pc')
        right_edge = arb_center + ds.quan(500, 'pc')
        obj = ds.arbitrary_grid(left_edge, right_edge, dims=(xlim,ylim,zlim))
        

        z_range_scaled = (0, 256)
        center_slice = np.log10(obj['flash', 'dens'][:, :, (int(pc2pixel(args.center_z_pc, x_y_z="z") * 256/1000) - z_range_scaled[0])].T[::])
        # center_slice = np.array(center_slice)
        
        img_root = ensure_dir(os.path.join(args.output_root, timestamp, "img"))
        mask_root = ensure_dir(os.path.join(args.output_root, timestamp, "mask"))

        # processing center slice
        img_path = os.path.join(img_root, f"{int(pc2pixel(args.center_z, x_y_z="z") * 256/1000)}.jpg")
        mask_path = os.path.join(mask_root, f"{int(pc2pixel(args.center_z, x_y_z="z") * 256/1000)}.png")
        cv2.imwrite(img_path, center_slice)

        volume_center = sam_and_save_mask(img_path, mask_path, input_box = args.bbox, input_point = [int(pc2pixel(args.center_x_pc, x_y_z="x") * 256/1000), int(pc2pixel(args.center_y_pc, x_y_z="y") * 256/1000)])

        # track up
        volume = volume_center
        z_scaled = int(pc2pixel(args.center_z, x_y_z="z") * 256/1000)
        while(volume > args.disappear_thres and z_scaled >= 0):
            z_scaled -= 1
            img_path = os.path.join(img_root, f"{z_scaled}.jpg")
            mask_path = os.path.join(mask_root, f"{z_scaled}.png")  
            volume = sam_and_save_mask(img_path, mask_path, input_box = args.bbox, input_point = [int(pc2pixel(args.center_x_pc, x_y_z="x") * 256/1000), int(pc2pixel(args.center_y_pc, x_y_z="y") * 256/1000)])
            

        # track down
        volume = volume_center
        z_scaled = int(pc2pixel(args.center_z, x_y_z="z") * 256/1000)
        while(volume > args.disappear_thres and z_scaled < 256):
            z_scaled += 1
            img_path = os.path.join(img_root, f"{z_scaled}.jpg")
            mask_path = os.path.join(mask_root, f"{z_scaled}.png")  
            volume = sam_and_save_mask(img_path, mask_path, input_box = args.bbox, input_point = [int(pc2pixel(args.center_x_pc, x_y_z="x") * 256/1000), int(pc2pixel(args.center_y_pc, x_y_z="y") * 256/1000)])

        # volume, kin_energy, therm_energy = SAM_segmentation()


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_root", help="The root directory to the hdf5 dataset")          # "/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc"
    parser.add_argument("--start_time_Myr", help="Specify the starting time (Myr)", default = 209, type = int)
    parser.add_argument("--end_time_Myr", help="Specify the starting time (Myr)", default = 246, type = int)   
    parser.add_argument("--interval", help="Specify the interval between timestamps", default = 10, type = int) 
    parser.add_argument("--file_prefix", help="sn34_smd132_bx5_pe300_hdf5_plt_cnt_0", default = "sn34_smd132_bx5_pe300_hdf5_plt_cnt_0")
    parser.add_argument("--center_x_pc", help="Specify the center position of SN in pc", type = int)            # 
    parser.add_argument("--center_y_pc", help="Specify the center position of SN in pc", type = int)            # 
    parser.add_argument("--center_z_pc", help="Specify the center position of SN in pc", type = int)            # 
    parser.add_argument("--bbox", help="Specify the center bbox of SB", type = list)            # [100, 128, 200, 228]
    parser.add_argument("--disappear_thres", help="Specify disappear iou threshold", default = 10, type = float) 
    parser.add_argument("--output_root", help="Path to output root", default = "../../Dataset/")    
    
    

  
    # python analysis/track_volume.py 
    
    args = parser.parse_args()
    main(args)
