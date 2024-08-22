import os
import numpy as np
import yt
import cv2 as cv
from matplotlib import pyplot as plt
from utils import *
import argparse

k = yt.physical_constants.kb
mu = 1.4
m_H = yt.physical_constants.mass_hydrogen


import matplotlib.pyplot as plt
import os

def plot_energy(timeMyrs, kinetic_energies, thermal_energies, total_energies, output_root):
    plt.figure(figsize=(12, 8))
    
    # Plotting with improved aesthetics
    plt.plot(timeMyrs, kinetic_energies, label='Kinetic Energy (erg)', color='#6A9C89', linestyle='dotted', linewidth=2, marker='o')
    plt.plot(timeMyrs, thermal_energies, label='Thermal Energy (erg)', color='#E1D7B7', linestyle='dotted', linewidth=2, marker='o')
    plt.plot(timeMyrs, total_energies, label='Total Energy (erg)', color='#CD5C08', linestyle='solid', linewidth=2.5, marker='o')
    
    # Logarithmic scale for the y-axis
    plt.yscale('log')
    
    # Adding labels and title
    plt.xlabel('Time (Myr)', fontsize=14)
    plt.ylabel('Energy (erg)', fontsize=14)
    plt.title('Energy vs. Time', fontsize=16)
    
    # Adding grid for better readability
    plt.grid(True, which="both", linestyle='--', linewidth=0.5, color='gray')
    
    # Customizing the legend
    plt.legend(fontsize=12, loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Adjusting layout for better spacing
    plt.tight_layout()
    
    # Save the figure to the specified output root
    plt.savefig(os.path.join(output_root, 'energy.png'), dpi=300)


def calc_energy(args, hdf5_filename, root_dir, timestamp):
    ds = yt.load(hdf5_filename)

    center = [0, 0, 0] * yt.units.pc    
    arb_center = ds.arr(center, 'code_length')
    xlim, ylim, zlim = args.pixel_boundary, args.pixel_boundary, args.pixel_boundary
    left_edge = arb_center + ds.quan(-500, 'pc')
    right_edge = arb_center + ds.quan(500, 'pc')
    obj = ds.arbitrary_grid(left_edge, right_edge, dims=(xlim,ylim,zlim))

    timestamp_energy = {'kinetic_energy': 0, 'thermal_energy': 0, 'total_energy': 0}

    mask_names = sorted(os.listdir(os.path.join(root_dir, str(timestamp))), key=lambda mask_name: int(mask_name.split('.')[0])) 

    for mask_name in mask_names:
        mask_img = cv.imread(os.path.join(root_dir, str(timestamp), mask_name), cv.IMREAD_GRAYSCALE)
        # coordinates = np.argwhere(mask_img == 255)
        mask_boolean = mask_img == 255

        z = int(mask_name.split('.')[0])
        # z = pixel2pc(int(mask_path.split(".")[-2].split("z")[-1]), x_y_z="z")

        temp = obj["flash", "temp"][:, :, z]
        n = obj["flash", "dens"][:, :, z] / (mu * m_H)

        rho = obj["flash", "dens"][:, :, z]
        v_sq = obj["flash", "velx"][:, :, z]**2 + obj["flash", "vely"][:, :, z]**2 + obj["flash", "velz"][:, :, z]**2

        cell_volume = obj["flash", "cell_volume"][:, :, z]

        kinetic_energy = (0.5 * rho * v_sq * cell_volume).to('erg')
        thermal_energy = ((3/2) * k * temp * n * cell_volume).to('erg')
        # total_energy = (kinetic_energy + thermal_energy).to('erg/cm**3')

        timestamp_energy['kinetic_energy'] += np.sum(kinetic_energy[mask_boolean])
        timestamp_energy['thermal_energy'] += np.sum(thermal_energy[mask_boolean])
        timestamp_energy['total_energy'] += np.sum(kinetic_energy[mask_boolean] + thermal_energy[mask_boolean])

        # for coord in coordinates:
        #     x, y = coord
        #     timestamp_energy['kinetic_energy'] += kinetic_energy[x, y] 
        #     timestamp_energy['thermal_energy'] += thermal_energy[x, y]
        #     timestamp_energy['total_energy'] += total_energy[x, y]


    return timestamp_energy




def main(args):
    timestamps = os.listdir(args.mask_root)
    timestamps = [int(timestamp) for timestamp in sorted(timestamps) if os.path.isdir(os.path.join(args.mask_root, timestamp))] 
    energy_data = {}

    for timestamp in timestamps:
        #DEBUG
        print(f"Processing {timestamp}")
        hdf5_filename = os.path.join(args.hdf5_root, f"{args.file_prefix}{timestamp}")
        timestamp_energy = calc_energy(args, hdf5_filename, args.mask_root, timestamp)
        energy_data[timestamp] = timestamp_energy


    # Plotting
    timestamps = list(energy_data.keys())
    kinetic_energies = [energy_data[timestamp]['kinetic_energy'] for timestamp in timestamps]
    thermal_energies = [energy_data[timestamp]['thermal_energy'] for timestamp in timestamps]
    total_energies = [energy_data[timestamp]['total_energy'] for timestamp in timestamps]
    timesMyrs = [timestamp2time_Myr(x) for x in list(energy_data.keys())] 

    plot_energy(timesMyrs, kinetic_energies, thermal_energies, total_energies, args.output_root)

    # Accumulated total energy
    print(f"Accumulated Kinetic Energy: {sum(kinetic_energies)} erg")
    print(f"Accumulated Thermal Energy: {sum(thermal_energies)} erg")
    print(f"Accumulated Total Energy: {sum(total_energies)} erg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_root", help="The root directory to the dataset")          # "../Dataset"
    parser.add_argument("--hdf5_root", help="The root directory to the dataset")
    parser.add_argument("--output_root", help="Path to output root", default = "../../Dataset/Isolated_case")
    parser.add_argument("--file_prefix", help="file prefix", default="sn34_smd132_bx5_pe300_hdf5_plt_cnt_0")
    parser.add_argument('-pixb', '--pixel_boundary', help='Input the pixel resolution', default = 256, type = int)
    
  
    # python analysis/calc_energy_map.py --mask_root /home/joy0921/Desktop/Dataset/Isloated_case/tmp --hdf5_root /home/joy0921/Desktop/Dataset/200_360/finer_time_200_360_original --output_root /home/joy0921/Desktop/Dataset/Isloated_case/tmp
    args = parser.parse_args()
    main(args)









