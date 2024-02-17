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


def plot_energy(timestamps, kinetic_energies, thermal_energies, total_energies, output_root):
    plt.figure(figsize=(10, 6))
    # plt.subplot(1, 2, 1)
    plt.plot(timestamps, kinetic_energies, label=f'Kinetic Energy (erg/cm³)')
    plt.plot(timestamps, thermal_energies, label=f'Thermal Energy (erg/cm³)')
    plt.plot(timestamps, total_energies, label=f'Total Energy (erg/cm³)')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Energy (erg/cm³)')
    plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(timestamps, volumes, label='Volume', color='purple')
    # plt.xlabel('Time')
    # plt.ylabel('Volume (pixels)')
    # plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_root, 'energy_chart.png'))

def calc_energy(hdf5_filename, root_dir, timestamp, xlim, ylim, zlim):
    ds = yt.load(hdf5_filename)

    center =  [0, 0, 0]*yt.units.pc
    arb_center = ds.arr(center,'code_length')
    left_edge = arb_center - ds.quan(int(xlim / 2),'pc')
    right_edge = arb_center + ds.quan(int(ylim / 2),'pc')
    obj = ds.arbitrary_grid(left_edge, right_edge, dims=[xlim, ylim, zlim])

    timestamp_energy = {'kinetic_energy': 0, 'thermal_energy': 0, 'total_energy': 0}

    masks = sorted(os.listdir(os.path.join(root_dir, str(timestamp))))

    for mask in masks:
        mask_img = cv.imread(os.path.join(root_dir, str(timestamp), mask), cv.IMREAD_GRAYSCALE)
        # coordinates = np.argwhere(mask_img == 255)
        mask_boolean = mask_img == 255

        z = pixel2pc(int(mask.split(".")[-2].split("z")[-1]), x_y_z="z")

        temp = obj["flash", "temp"][:, :, z]
        n = obj["flash", "dens"][:, :, z] / (mu * m_H)

        rho = obj["flash", "dens"][:, :, z]
        v_sq = obj["flash", "velx"][:, :, z]**2 + obj["flash", "vely"][:, :, z]**2 + obj["flash", "velz"][:, :, z]**2

        cell_volume = obj["flash", "cell_volume"][:, :, z]

        kinetic_energy = (0.5 * rho * v_sq * cell_volume).to('erg')
        thermal_energy = ((3/2) * k * temp * n * cell_volume).to('erg/cm**3')
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
        timestamp_energy = calc_energy(hdf5_filename, args.mask_root, timestamp, args.xlim, args.ylim, args.zlim)
        energy_data[timestamp] = timestamp_energy


    # Plotting
    timestamps = list(energy_data.keys())
    kinetic_energies = [energy_data[timestamp]['kinetic_energy'] for timestamp in timestamps]
    thermal_energies = [energy_data[timestamp]['thermal_energy'] for timestamp in timestamps]
    total_energies = [energy_data[timestamp]['total_energy'] for timestamp in timestamps]

    plot_energy(timestamps, kinetic_energies, thermal_energies, total_energies, args.output_root)

    # Accumulated total energy
    print(f"Accumulated Kinetic Energy: {sum(kinetic_energies)} erg/cm³")
    print(f"Accumulated Thermal Energy: {sum(thermal_energies)} erg/cm³")
    print(f"Accumulated Total Energy: {sum(total_energies)} erg/cm³")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_root", help="The root directory to the dataset")          # "../Dataset"
    parser.add_argument("--hdf5_root", help="The root directory to the dataset")
    parser.add_argument("--output_root", help="Path to output root", default = "../../Dataset/Isolated_case")
    parser.add_argument("--file_prefix", help="file prefix", default="sn34_smd132_bx5_pe300_hdf5_plt_cnt_0")
    parser.add_argument("--xlim", help="Input xlim", default = 1000, type = int)                                         # 1000 
    parser.add_argument("--ylim", help="Input ylim", default = 1000, type = int)                                         # 1000  
    parser.add_argument("--zlim", help="Input zlim", default = 1000, type = int)                                         # 1000
    
  
    # python analysis/calc_energy_map.py --mask_root /home/joy0921/Desktop/Dataset/Isloated_case/tmp --hdf5_root /home/joy0921/Desktop/Dataset/200_360/finer_time_200_360_original --output_root /home/joy0921/Desktop/Dataset/Isloated_case/tmp
    args = parser.parse_args()
    main(args)









