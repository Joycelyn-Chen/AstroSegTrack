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
import math
from utils import *

limits = [0.0,      2.51e1,  3.98e1,  2.00e2,  1.0e3,   3.16e3,  6.31e3,  1.0e4,   1.70e4,   3.98e4, 7.94e4,  2.51e5,  5.62e5,  1.78e6, 2.75e6,   3.16e7]
powers = [3.885,    1.50,    0.997,   0.431,   0.352,   0.152,   0.396,   13.8,    -0.216,   2.0,    0.01,    -2.0,    0.01,    -2.95,  -0.33,    0.50]
coef =   [1.095e-32,2.39e-29,1.52e-28,3.06e-27,5.28e-27,2.64e-26,3.13e-27,7.63e-81,1.479e-21,1.0e-31,5.50e-22,3.98e-11,1.15e-22,3.89e-4,5.188e-21,3.090e-27]

log_limits = np.log(limits)
log_powers = np.log(powers)
log_coef = np.log(coef)

piecewise_interp = interp1d(log_limits, log_coef + np.multiply(log_powers, log_limits[:-1]), kind='linear', fill_value="extrapolate")

low_x0, low_y0, low_w, low_h, bottom_z, top_z = -500, -500, 1000, 1000, -500, 500
k = yt.physical_constants.kb
mu = 1.4
m_H = yt.physical_constants.mass_hydrogen
xlim = 256
ylim = 256
zlim= 256
center = [0, 0, 0] * yt.units.pc
z_range_scaled = (0, 256)

epsilon = 0.05
G_0 = 1.7
h_pe = 300


DEBUG = True



def otsu_and_save_mask(image_path, output_path, input_point):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = apply_otsus_thresholding(image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Check each component's bounding box to find the target mask
    for i in range(3, num_labels):  # Starting from 1 to ignore the background
        x, y, w, h, area = stats[i]
        
        # Check if the center point is within the bounding box
        if SN_center_in_bubble(input_point[0], input_point[1], x, y, w, h):     # and area < 10 * 427
            # If yes, fill the target_mask with this component
            binary_mask = labels == i
            target_mask = np.where(labels == i, 255, 0).astype('uint8')

            # Save the target mask
            cv2.imwrite(output_path, target_mask)
            
            break 
    if area is not None:
        return area, binary_mask
    return 0, None
    # area = np.sum(binary_mask) / 255
    # return area



def associate_slices_within_cube(obj, center_mask, img_root, mask_root, z_scaled, disappear_thres, direction, points, half_radius = 50):     #directom: -1 up, -1 down
    area = disappear_thres
    incr = 0
    half_volume = 0
    half_kinetic = 0
    half_thermal = 0
    half_total = 0
    half_heating = 0
    half_cooling = 0
    tmp_mask = center_mask
    
    while(area >= disappear_thres and incr <= half_radius):
        incr += 1
        z_scaled += direction * 1
        img_path = os.path.join(img_root, f"{z_scaled}.jpg")
        mask_path = os.path.join(mask_root, f"{z_scaled}.png")  
        
        # store next image
        next_slice = np.log10(obj['flash', 'dens'][:, :, z_scaled].T[::])
        next_slice_norm = ((next_slice - np.min(next_slice)) / (np.max(next_slice) - np.min(next_slice)) ) * 255 
        cv2.imwrite(img_path, next_slice_norm) 
        
        image = read_image_grayscale(img_path)
        binary_image = apply_otsus_thresholding(image)
        num_labels, labels, stats, _ = find_connected_components(binary_image)

        no_match = True
        for label in range(2, num_labels):
            current_mask = labels == label

            if compute_iou(current_mask, tmp_mask) >= 0.5:      # if found a match in this slice
                tmp_mask = current_mask 
                cv2.imwrite(mask_path, current_mask * 255)

                # log volume for each slice
                area = stats[label, cv2.CC_STAT_AREA]
                half_volume += area
                no_match = False
                kinetic_energy, thermal_energy, total_energy, heating_rate, cooling_rate = calc_energy(obj, mask_path)

                half_kinetic += kinetic_energy
                half_thermal += thermal_energy
                half_total += total_energy
                half_heating += heating_rate
                half_cooling += cooling_rate
                half_volume += area

                if (DEBUG):
                    print("Z: {}\tArea: {}".format(z_scaled, area))
                break       # Moving to the next slice
        
        # If can't find any match in this slice, then move on to the next phase
        if no_match:
            break

    return half_volume, half_kinetic, half_thermal, half_total, half_heating, half_cooling


p = PiecewisePowerlaw(limits=limits, powers=powers, coefficients=coef, norm=False)

def cooling(Tinp, dimensions=False):
    if dimensions:
        T = np.array(Tinp / yt.units.K)
        unit = yt.units.erg * yt.units.cm**3.0 * yt.units.s**(-1.0)
    else:
        T = np.array(Tinp)
        unit = 1
    return p(T) * unit

def calc_energy(obj, mask_path):
    if(DEBUG):
        print("Calculating 3 energies...\n")

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # coordinates = np.argwhere(mask_img == 255)
    mask_boolean = mask_img == 255

    z = int(mask_path.split("/")[-1].split(".")[-2])            # z in pc

    temp = obj["flash", "temp"][:, :, z]
    n = obj["flash", "dens"][:, :, z] / (mu * m_H)

    rho = obj["flash", "dens"][:, :, z]
    v_sq = obj["flash", "velx"][:, :, z]**2 + obj["flash", "vely"][:, :, z]**2 + obj["flash", "velz"][:, :, z]**2

    cell_volume = obj["flash", "cell_volume"][:, :, z]


    # Heating Rate
    temp_roi = np.where(mask_boolean, temp, np.nan)
    z_pc = pixel2pc(z / 0.256)          # convert 256pixel-z to z in pc
    heating_gamma = np.where(temp_roi > 20000, 0, epsilon * G_0 * np.exp(-np.abs(z_pc) / h_pe) * 1e-24)        # Calculate heating_gamma based on temperature
    heating_gamma_n = np.multiply(heating_gamma, n)                                                         # Multiply heating_gamma with n
    heating_rate = np.sum(heating_gamma_n)

    # Cooling Rate
    cooling_rate = np.sum(np.multiply(cooling(temp_roi, dimensions=False), n ** 2)) 

    kinetic_energy = (0.5 * rho * v_sq * cell_volume).to('erg')
    thermal_energy = ((3/2) * k * temp * n * cell_volume).to('erg')
    total_energy = (kinetic_energy + thermal_energy).to('erg')

    kinetic_energy_sum = np.sum(kinetic_energy[mask_boolean])
    thermal_energy_sum = np.sum(thermal_energy[mask_boolean])
    total_energy = kinetic_energy_sum + thermal_energy_sum
    return kinetic_energy_sum, thermal_energy_sum, total_energy, heating_rate, cooling_rate


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

def trace_first_timestamp(args, timestamp, timestamp_info):
    ds = yt.load(os.path.join(args.hdf5_root, '{}{}'.format(args.file_prefix, timestamp)))

    arb_center = ds.arr(center, 'code_length')
    left_edge = arb_center + ds.quan(-500, 'pc')
    right_edge = arb_center + ds.quan(500, 'pc')
    obj = ds.arbitrary_grid(left_edge, right_edge, dims=(xlim,ylim,zlim))
    

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

    area_center, center_mask = otsu_and_save_mask(img_path, mask_path, input_point = points)
    timestamp_info[timestamp] = {}
    timestamp_info[timestamp]['volume'] = area_center
    timestamp_info[timestamp]['kinetic'] = 0        # TODO: calc energy for center slice
    timestamp_info[timestamp]['thermal'] = 0
    timestamp_info[timestamp]['total'] = 0
    timestamp_info[timestamp]['heating'] = 0
    timestamp_info[timestamp]['cooling'] = 0

    if(DEBUG):
        print("Center area: {}".format(area_center))
        print("Tracking up for timestamp {}".format(timestamp))
    
    # track up
    z_scaled = int(pc2pixel(args.center_z_pc, x_y_z="z") * 256/1000)

    half_volume, half_kinetic, half_thermal, half_total, half_heating, half_cooling = associate_slices_within_cube(obj, center_mask, img_root, mask_root, z_scaled - 1, disappear_thres = args.disappear_thres, direction = -1, half_radius = 100, points = points)
    timestamp_info[timestamp]['volume'] += half_volume
    timestamp_info[timestamp]['kinetic'] += half_kinetic        
    timestamp_info[timestamp]['thermal'] += half_thermal
    timestamp_info[timestamp]['total'] += half_total
    timestamp_info[timestamp]['heating'] += half_heating
    timestamp_info[timestamp]['cooling'] += half_cooling

    if(DEBUG):
        print("Tracking down for timestamp {}".format(timestamp))
    # track down
    half_volume, half_kinetic, half_thermal, half_total, half_heating, half_cooling =  associate_slices_within_cube(obj, center_mask, img_root, mask_root, z_scaled + 1, disappear_thres = args.disappear_thres, direction = +1, half_radius = 100, points = points)
    timestamp_info[timestamp]['volume'] += half_volume
    timestamp_info[timestamp]['kinetic'] += half_kinetic        
    timestamp_info[timestamp]['thermal'] += half_thermal
    timestamp_info[timestamp]['total'] += half_total
    timestamp_info[timestamp]['heating'] += half_heating
    timestamp_info[timestamp]['cooling'] += half_cooling

    if(DEBUG):
        print(timestamp_info)
    
    return center_mask, timestamp_info 

def associate_next_timestamp(args, timestamp, timestamp_info):
    ds = yt.load(os.path.join(args.hdf5_root, '{}{}'.format(args.file_prefix, timestamp)))

    arb_center = ds.arr(center, 'code_length')
    left_edge = arb_center + ds.quan(-500, 'pc')
    right_edge = arb_center + ds.quan(500, 'pc')
    obj = ds.arbitrary_grid(left_edge, right_edge, dims=(xlim,ylim,zlim))
    
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

    area_center, center_mask = otsu_and_save_mask(img_path, mask_path, input_point = points)
    timestamp_info[timestamp]['volume'] = 0
    timestamp_info[timestamp]['kinetic'] = 0        # TODO: calc energy for center slice
    timestamp_info[timestamp]['thermal'] = 0
    timestamp_info[timestamp]['total'] = 0
    timestamp_info[timestamp]['heating'] = 0
    timestamp_info[timestamp]['cooling'] = 0

    if(DEBUG):
        print("Center area: {}".format(area_center))
        print("Tracking up for timestamp {}".format(timestamp))
    
    # track up
    z_scaled = int(pc2pixel(args.center_z_pc, x_y_z="z") * 256/1000)
    
    half_volume, half_kinetic, half_thermal, half_total, half_heating, half_cooling = associate_slices_within_cube(obj, center_mask, img_root, mask_root, z_scaled, disappear_thres = args.disappear_thres, direction = -1, half_radius = 100, points = points)
    timestamp_info[timestamp]['volume'] += half_volume
    timestamp_info[timestamp]['kinetic'] += half_kinetic        
    timestamp_info[timestamp]['thermal'] += half_thermal
    timestamp_info[timestamp]['total'] += half_total
    timestamp_info[timestamp]['heating'] += half_heating
    timestamp_info[timestamp]['cooling'] += half_cooling


    if(DEBUG):
        print("Tracking down for timestamp {}".format(timestamp))
    # track down
    half_volume, half_kinetic, half_thermal, half_total, half_heating, half_cooling =  associate_slices_within_cube(obj, center_mask, img_root, mask_root, z_scaled + 1, disappear_thres = args.disappear_thres, direction = +1, half_radius = 100, points = points)
    timestamp_info[timestamp]['volume'] += half_volume
    timestamp_info[timestamp]['kinetic'] += half_kinetic        
    timestamp_info[timestamp]['thermal'] += half_thermal
    timestamp_info[timestamp]['total'] += half_total
    timestamp_info[timestamp]['heating'] += half_heating
    timestamp_info[timestamp]['cooling'] += half_cooling

    return center_mask


def segment_and_accumulate_areas(args, start_timestamp, end_timestamp, timestamp_info):
    timestamps = range(start_timestamp + args.interval, end_timestamp, args.interval)  
    blob_disappeared = False

    # trace the first center timestamp
    # retrieve mask, volume, energy
    # TODO: modify trace_first_timestamp
    center_mask, timestamp_info = trace_first_timestamp(args, start_timestamp, timestamp_info)
    previous_mask = center_mask

    #DEBUG
    if (DEBUG):
        print(f"Done tracing first timestamp {start_timestamp}...")


    for timestamp in timestamps:
        if blob_disappeared:
            break
        
        # initialization 
        timestamp_info[timestamp] = {}

        center_mask = associate_next_timestamp(args, timestamp, timestamp_info)
        if center_mask is None or compute_iou(previous_mask, center_mask) < 0.3: #disappear_thres:
            blob_disappeared = True
            continue
        previous_mask = center_mask

        
        #DEBUG
        if(DEBUG):
            print(f"Done tracing {timestamp}... volume = {timestamp_info[timestamp]['volume']}")
            print(timestamp_info)
    
    return timestamp_info





def main(args):
    start_timestamp = time_Myr2timestamp(args.start_time_Myr)
    end_timestamp = time_Myr2timestamp(args.end_time_Myr) + 1
    timestamp_info = {}
    

    timestamp_info = segment_and_accumulate_areas(args, start_timestamp, end_timestamp, timestamp_info)
    
    # TODO: plot the energy and volume chart
    # plot_accumulated_volumes(accumulated_volumes, mask_dir_root)
    
    
    # with open(os.path.join(mask_dir_root, "volume.json"), "w") as f:
    #     json.dump(accumulated_volumes_int, f)

    
  

    with open(os.path.join(args.output_root, 'timestamp_info.json'), f'{args.info_mode}') as convert_file: 
        convert_file.write(str(timestamp_info))


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
    parser.add_argument("--info_mode", help = "Whether you're writing (w) or appending (a) into the info.json file")
    
    

  
    # python analysis/superbubble_segmentation.py --hdf5_root /home/joy0921/Desktop/Dataset/SB230/HDF5 --start_time_Myr 209 --end_time_Myr 211 --center_x_pc 85 --center_y_pc 196 --center_z_pc 53 --output_root ../Dataset/SB230 --info_mode w   
    args = parser.parse_args()
    main(args)
