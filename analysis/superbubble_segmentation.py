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
k = yt.physical_constants.kb
mu = 1.4
m_H = yt.physical_constants.mass_hydrogen


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = './checkpoints/sam_vit_h_4b8939.pth'
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

    area = np.sum(best_mask)
    return area

def otsu_and_save_mask(image_path, output_path, input_point):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = apply_otsus_thresholding(image)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Check each component's bounding box to find the target mask
    for i in range(3, num_labels):  # Starting from 1 to ignore the background
        x, y, w, h, area = stats[i]
        
        # Check if the center point is within the bounding box
        if (x <= input_point[0] < x + w) and (y <= input_point[1] < y + h and area < 10 * 427):
            # If yes, fill the target_mask with this component
            binary_mask[labels == i] = 255
            target_mask = np.where(labels == i, 255, 0).astype('uint8')

            # Save the target mask
            cv2.imwrite(output_path, target_mask)
            
            break 
    if area is not None:
        return area 
    return 0
    # area = np.sum(binary_mask) / 255
    # return area



def associate_slices_within_cube(obj, img_root, mask_root, z_scaled, disappear_thres, direction, points, half_radius = 25):     #directom: -1 up, -1 down
    area = disappear_thres
    incr = 0
    half_volume = 0

    while(area >= disappear_thres and incr <= half_radius):
        incr += 1
        z_scaled += direction * 1
        img_path = os.path.join(img_root, f"{z_scaled}.jpg")
        mask_path = os.path.join(mask_root, f"{z_scaled}.png")  
        
        # store next image
        next_slice = np.log10(obj['flash', 'dens'][:, :, z_scaled].T[::])
        next_slice_norm = ((next_slice - np.min(next_slice)) / (np.max(next_slice) - np.min(next_slice)) ) * 255 
        cv2.imwrite(img_path, next_slice_norm) 
        
        area = otsu_and_save_mask(img_path, mask_path, input_point = points)
        kinetic_energy, thermal_energy, total_energy = calc_energy(obj, mask_path)

        half_kinetic += kinetic_energy
        half_thermal += thermal_energy
        half_total += total_energy
        half_volume += area

        if (DEBUG):
            print("Z: {}\tArea: {}".format(z_scaled, area))
    return half_volume, half_kinetic, half_thermal, half_total

def calc_energy(obj, mask_path):
    if(DEBUG):
        print("Calculating 3 energies...\n")

    mask_img = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    # coordinates = np.argwhere(mask_img == 255)
    mask_boolean = mask_img == 255

    z = int(mask_path.split(".")[-2])

    temp = obj["flash", "temp"][:, :, z]
    n = obj["flash", "dens"][:, :, z] / (mu * m_H)

    rho = obj["flash", "dens"][:, :, z]
    v_sq = obj["flash", "velx"][:, :, z]**2 + obj["flash", "vely"][:, :, z]**2 + obj["flash", "velz"][:, :, z]**2

    cell_volume = obj["flash", "cell_volume"][:, :, z]

    kinetic_energy = (0.5 * rho * v_sq * cell_volume).to('erg')
    thermal_energy = ((3/2) * k * temp * n * cell_volume).to('erg')
    total_energy = (kinetic_energy + thermal_energy).to('erg/cm**3')

    kinetic_energy_sum = np.sum(kinetic_energy[mask_boolean])
    thermal_energy_sum = np.sum(thermal_energy[mask_boolean])
    total_energy = kinetic_energy_sum + thermal_energy_sum
    return kinetic_energy_sum, thermal_energy_sum, total_energy


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
    timestamp_info = {}

    for timestamp in range(start_timestamp, end_timestamp, args.interval):
        # initialization 
        timestamp_volume = 0

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
        center_slice_norm = ((center_slice - np.min(center_slice)) / (np.max(center_slice) - np.min(center_slice)) ) * 255 
        # center_slice = np.array(center_slice)
        
        img_root = ensure_dir(os.path.join(args.output_root, str(timestamp), "img"))
        mask_root = ensure_dir(os.path.join(args.output_root, str(timestamp), "mask"))

        # processing center slice
        img_path = os.path.join(img_root, f"{int(pc2pixel(args.center_z_pc, x_y_z='z') * 256/1000)}.jpg")
        mask_path = os.path.join(mask_root, f"{int(pc2pixel(args.center_z_pc, x_y_z='z') * 256/1000)}.png")
        cv2.imwrite(img_path, center_slice_norm)

        if(DEBUG):
            print("Processing image: {}".format(img_path))
        
        points = [int(pc2pixel(args.center_x_pc, x_y_z="x") * 256/1000), int(pc2pixel(args.center_y_pc, x_y_z="y") * 256/1000) + 100]

        area_center = otsu_and_save_mask(img_path, mask_path, input_point = points)
        timestamp_info[timestamp]['volume'] = area_center
        timestamp_info[timestamp]['kinetic'] = 0        # TODO: calc energy for center slice
        timestamp_info[timestamp]['thermal'] = 0
        timestamp_info[timestamp]['total'] = 0

        if(DEBUG):
            print("Center area: {}".format(area_center))
            print("Tracking up for timestamp {}".format(timestamp))
        
        # track up
        area = area_center
        z_scaled = int(pc2pixel(args.center_z_pc, x_y_z="z") * 256/1000)
        
        half_volume, half_kinetic, half_thermal, half_total = associate_slices_within_cube(obj, img_root, mask_root, z_scaled - 1, disappear_thres = args.disappear_thres, direction = -1, half_radius = 25, points = points)
        timestamp_info[timestamp]['volume'] += half_volume
        timestamp_info[timestamp]['kinetic'] += half_kinetic        
        timestamp_info[timestamp]['thermal'] += half_thermal
        timestamp_info[timestamp]['total'] += half_total

        if(DEBUG):
            print("Tracking down for timestamp {}".format(timestamp))
        # track down
        half_volume, half_kinetic, half_thermal, half_total =  associate_slices_within_cube(obj, img_root, mask_root, z_scaled + 1, disappear_thres = args.disappear_thres, direction = +1, half_radius = 25, points = points)
        timestamp_info[timestamp]['volume'] += half_volume
        timestamp_info[timestamp]['kinetic'] += half_kinetic        
        timestamp_info[timestamp]['thermal'] += half_thermal
        timestamp_info[timestamp]['total'] += half_total

        print(timestamp_info)
        
        # volume, kin_energy, therm_energy = SAM_segmentation()

        
        
        # TODO: still need to document volume, energy and write them into csv file 


    



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
    parser.add_argument("--bbox", help="Specify the center bbox of SB", type = list, default = [120, 148, 180, 208])            # [100, 128, 200, 228]
    parser.add_argument("--disappear_thres", help="Specify disappear area threshold (pixel)", default = 10, type = float) 
    parser.add_argument("--output_root", help="Path to output root", default = "../../Dataset/")    
    
    

  
    # python analysis/superbubble_segmentation.py --hdf5_root /home/joy0921/Desktop/Dataset/SB230/HDF5 --start_time_Myr 209 --end_time_Myr 209 --center_x_pc 85 --center_y_pc 196 --center_z_pc 53 --output_root ../Dataset/SB230    
    args = parser.parse_args()
    main(args)
