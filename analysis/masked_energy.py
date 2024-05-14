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

# coefficient for cooling
limits = [0.0,      2.51e1,  3.98e1,  2.00e2,  1.0e3,   3.16e3,  6.31e3,  1.0e4,   1.70e4,   3.98e4, 7.94e4,  2.51e5,  5.62e5,  1.78e6, 2.75e6,   3.16e7, np.infty]
powers = [3.885,    1.50,    0.997,   0.431,   0.352,   0.152,   0.396,   13.8,    -0.216,   2.0,    0.01,    -2.0,    0.01,    -2.95,  -0.33,    0.50]
coef =   [1.095e-32,2.39e-29,1.52e-28,3.06e-27,5.28e-27,2.64e-26,3.13e-27,7.63e-81,1.479e-21,1.0e-31,5.50e-22,3.98e-11,1.15e-22,3.89e-4,5.188e-21,3.090e-27]

# parameters for grid conversion
low_x0, low_y0, low_w, low_h, bottom_z, top_z = -500, -500, 1000, 1000, -500, 500
xlim = 256
ylim = 256
zlim= 256
center = [0, 0, 0] * yt.units.pc
z_range_scaled = (0, 256)

# kinetic/thermal energy constants
k = yt.physical_constants.kb
mu = 1.4
m_H = yt.physical_constants.mass_hydrogen

# heating/cooling constants
epsilon = 0.05
G_0 = 1.7
h_pe = 300
pc2cm = 3.086e18        # 1 pc in cm
Myr2sec = 3.1536e13     # 1 Myr in sec


# debugging
DEBUG = False

p = PiecewisePowerlaw(limits=limits, powers=powers, coefficients=coef, norm=False)


def cooling_func(Tinp, dimensions=False):
    if dimensions:
        T = np.array(Tinp / yt.units.K)
        unit = yt.units.erg * yt.units.cm**3.0 * yt.units.s**(-1.0)
    else:
        T = np.array(Tinp)
        unit = 1
    return p(T) * unit

def energy_integral(energy, volume):
    # volume in pixel
    return energy * volume * (3.9*pc2cm) ** 3 * Myr2sec

def calc_energy(obj, mask_path, timestamp_info, timestamp):

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # coordinates = np.argwhere(mask_img == 255)
    mask_boolean = mask_img == 255
    
    mask_area = np.count_nonzero(mask_boolean)

    z = int(mask_path.split("/")[-1].split(".")[-2])            # z in 256-pixel

    temp = obj["flash", "temp"][:, :, z]
    n = obj["flash", "dens"][:, :, z] / (mu * m_H)

    rho = obj["flash", "dens"][:, :, z]
    v_sq = obj["flash", "velx"][:, :, z]**2 + obj["flash", "vely"][:, :, z]**2 + obj["flash", "velz"][:, :, z]**2

    cell_volume = obj["flash", "cell_volume"][:, :, z]


    # Heating Rate
    temp_roi = np.where(mask_boolean, temp, 20001)          # TODO: could be a concern, if there's another analysis goal, since I'm setting the uninterested region to 20001
    
    
    z_pc = (z - 128) * 3.9          # converting z coordinate to pc
    # z_pc = pixel2pc(z, "z")  / 0.256          # convert 256pixel-z to z in pc
    if(DEBUG):
        print("z_pc: ", z_pc)
    
    heating_gamma = np.where(temp_roi > 20000, 0, epsilon * G_0 * np.exp(-np.abs(z_pc) / h_pe) * 1e-24)        # Calculate heating_gamma based on temperature
    if(DEBUG):          # getting a sense of the heating portion within the segmented region
        print("non-zeros:", np.count_nonzero(heating_gamma), "out of: ", heating_gamma.size, "= ", (np.count_nonzero(heating_gamma)/heating_gamma.size) * 100, "%")
        
    heating_gamma_n = np.multiply(heating_gamma, n)                                                         # Multiply heating_gamma with n
    heating = np.sum(energy_integral(heating_gamma_n, mask_area))


    # Cooling Rate
    temp_roi = np.where(mask_boolean, temp, np.nan)             # resetting temp from the region of interest, cause I mess with the one from above
    cooling = np.sum(energy_integral(np.multiply(cooling_func(temp_roi, dimensions=True), n ** 2), mask_area)) 

    # kinetic/thermal energy calculation
    kinetic_energy = (0.5 * rho * v_sq * cell_volume).to('erg')
    thermal_energy = ((3/2) * k * temp * n * cell_volume).to('erg')
    total_energy = (kinetic_energy + thermal_energy).to('erg')

    kinetic_sum = np.sum(kinetic_energy[mask_boolean])
    thermal_sum = np.sum(thermal_energy[mask_boolean])
    total_energy = kinetic_sum + thermal_sum
    
    return energy_doc(timestamp_info, timestamp, mask_area, kinetic_sum, thermal_sum, total_energy, heating, cooling) 

def read_mask_filenames(dataset_root, timestamp):
    mask_root = os.path.join(dataset_root, str(timestamp), "mask") 

    return sorted([os.path.join(mask_root, file) for file in os.listdir(mask_root)])

def energy_doc(timestamp_info, timestamp, mask_area, kinetic_sum, thermal_sum, total_energy, heating, cooling):
    if(timestamp not in timestamp_info):
        timestamp_info[timestamp] = {}
        timestamp_info[timestamp]['volume'] = mask_area
        timestamp_info[timestamp]['kinetic'] = kinetic_sum        
        timestamp_info[timestamp]['thermal'] = thermal_sum
        timestamp_info[timestamp]['total'] = total_energy
        timestamp_info[timestamp]['heating'] = heating
        timestamp_info[timestamp]['cooling'] = cooling
    else:
        timestamp_info[timestamp]['volume'] += mask_area
        timestamp_info[timestamp]['kinetic'] += kinetic_sum        
        timestamp_info[timestamp]['thermal'] += thermal_sum
        timestamp_info[timestamp]['total'] += total_energy
        timestamp_info[timestamp]['heating'] += heating
        timestamp_info[timestamp]['cooling'] += cooling
    
    print(timestamp_info[timestamp])
    return timestamp_info

def plot_energy(timestamp_info, timestamps, output_root):
    volume = []
    kinetic = []
    thermal = []
    total = []
    heating = []
    cooling = []
    

    for timestamp in timestamps:
        volume.append(timestamp_info[timestamp]['volume'])
        kinetic.append(timestamp_info[timestamp]['kinetic'])
        thermal.append(timestamp_info[timestamp]['thermal'])
        total.append(timestamp_info[timestamp]['total'])
        heating.append(timestamp_info[timestamp]['heating'])
        heating_values = np.array([h.value for h in heating])
        
        cooling.append(timestamp_info[timestamp]['cooling'])
        cooling_values = np.array([c.value for c in cooling])


    # convert timstamps to time_Myr
    time_Myr = [timestamp2time_Myr(timestamp) for timestamp in timestamps]  # 209 ~ 231

    # supernovae energy injection
    one_hot_explosion = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    E_sn = [num * 1e51 for num in one_hot_explosion]
    
    fig, ax1 = plt.subplots()
    ax1.plot(time_Myr, kinetic, label='Kinetic Energy (erg)', color = 'wheat')    #
    ax1.plot(time_Myr, thermal, label='Thermal Energy (erg)', color = 'xkcd:beige')    # 
    ax1.plot(time_Myr, total, label='Total Energy (erg)', color = 'xkcd:tan')        # gold
    ax1.plot(time_Myr, E_sn, 'r+')
    ax1.set_yscale('log')
    ax1.set_xlabel('Time (Myr)')
    ax1.set_ylabel('Energy (erg)')
    
    
    
    ax2 = ax1.twinx()
    ax2.plot(time_Myr, heating_values, label = 'Heating (erg)', color = 'lightblue')   #red, tomato
    ax2.plot(time_Myr, cooling_values, label = 'Cooling (erg)', color = 'xkcd:azure')  #powerblue
    # ax2.plot(timestamps, E_sn, label = "E_sn", color = "red")
    ax2.plot(time_Myr, np.abs(heating_values - cooling_values ), label = "H - C + E_sn", color = "xkcd:blue")
    ax2.set_yscale('log')
    ax2.set_ylabel('heating and cooling (erg)')
    
    # fig.tight_layout()
    fig.legend()
    plt.savefig(os.path.join(output_root, 'energy_chart.png'))
    plt.show()

def main(args):
    start_timestamp = time_Myr2timestamp(args.start_time_Myr)
    end_timestamp = time_Myr2timestamp(args.end_time_Myr) + 1
    timestamp_info = {}
    
    # for each timestamp, read the mask for each slice, then calculate energy
    for timestamp in range(start_timestamp, end_timestamp + 1, args.interval):    
        ds = yt.load(os.path.join(args.hdf5_root, '{}{}'.format(args.file_prefix, timestamp)))
        arb_center = ds.arr(center, 'code_length')
        left_edge = arb_center + ds.quan(-500, 'pc')
        right_edge = arb_center + ds.quan(500, 'pc')
        obj = ds.arbitrary_grid(left_edge, right_edge, dims=(xlim,ylim,zlim))
        
        # reading masks
        mask_files = read_mask_filenames(args.dataset_root, timestamp)
        for mask in mask_files:
            timestamp_info = calc_energy(obj, mask, timestamp_info, timestamp)
            
            
    timestamps = list(range(start_timestamp, end_timestamp + 1, args.interval))
    plot_energy(timestamp_info, timestamps, args.dataset_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_root", help="The root directory to the hdf5 dataset")          # "/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc"
    parser.add_argument("--start_time_Myr", help="Specify the starting time (Myr)", default = 209, type = int)
    parser.add_argument("--end_time_Myr", help="Specify the starting time (Myr)", default = 231, type = int)   
    parser.add_argument("--interval", help="Specify the interval between timestamps", default = 10, type = int) 
    parser.add_argument("--file_prefix", help="sn34_smd132_bx5_pe300_hdf5_plt_cnt_0", default = "sn34_smd132_bx5_pe300_hdf5_plt_cnt_0")
    parser.add_argument("--dataset_root", help="Path to output root", default = "../../Dataset/SB230")    
    

    # python analysis/masked_energy.py --hdf5_root /home/joy0921/Desktop/Dataset/SB230/HDF5 --start_time_Myr 209 --end_time_Myr 231 --dataset_root ../Dataset/SB230   
    args = parser.parse_args()
    main(args)
