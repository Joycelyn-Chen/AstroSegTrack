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
    plt.plot(timestamps, kinetic_energies, label=f'Kinetic Energy (erg)')
    plt.plot(timestamps, thermal_energies, label=f'Thermal Energy (erg)')
    plt.plot(timestamps, total_energies, label=f'Total Energy (erg)')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Energy (erg)')
    plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(timestamps, volumes, label='Volume', color='purple')
    # plt.xlabel('Time')
    # plt.ylabel('Volume (pixels)')
    # plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_root, 'energy_chart.png'))

def calc_energy(hdf5_filename, root_dir, timestamp, xlim, ylim, zlim, output_dir):
    ds = yt.load(hdf5_filename)

    center = [0, 0, 0] * yt.units.pc
    arb_center = ds.arr(center, 'code_length')
    left_edge = arb_center - ds.quan(int(xlim / 2), 'pc')
    right_edge = arb_center + ds.quan(int(ylim / 2), 'pc')
    obj = ds.arbitrary_grid(left_edge, right_edge, dims=[xlim, ylim, zlim])
    
    
    

    timestamp_energy = {'kinetic_energy': 0, 'thermal_energy': 0, 'total_energy': 0}

    masks = sorted(os.listdir(os.path.join(root_dir, str(timestamp))))
    center_mask_path = masks[(len(masks) // 2)]  # Ensure integer division

    # Initialize kinetic and thermal energy maps
    # kinetic_energy_map = np.zeros((ylim, xlim))
    # thermal_energy_map = np.zeros((ylim, xlim))
    kinetic_energy_map = yt.YTArray(np.zeros((1000, 1000, 1))) * yt.units.erg #/ yt.units.cm**3
    thermal_energy_map = yt.YTArray(np.zeros((1000, 1000, 1))) * yt.units.erg #/ yt.units.cm**3

    kinetic_map = yt.YTArray(np.zeros((1000, 1000, 1))) * yt.units.erg / yt.units.cm**3
    # cell_volume = 1.0000000000000004*yt.units.pc**3    #2.9379989445851796e+55
    k = yt.physical_constants.kb

    center_z = 409


    img = np.log10(obj["flash", "dens"][:,:,center_z])       # .T[::-1]
    normalizedImg = ((img - np.min(img)) / (np.max(img) - np.min(img)) ) * 255 
    cv.imwrite(os.path.join(output_dir, 'dens.png'), normalizedImg)

    rho = obj["flash", "dens"][:,:,center_z]
    v_sq = obj["flash", "velx"][:,:,center_z]**2 + obj["flash", "vely"][:,:,center_z]**2 + obj["flash", "velz"][:,:,center_z]**2
    kinetic_map[:, :, 0] = ((1/2) * rho * v_sq ).to(kinetic_map.units)

            
    plt.imshow(np.log10(kinetic_map), cmap = 'RdBu', vmin = -16, vmax = -9)
    plt.colorbar()
    plt.savefig(f"./analysis/graphs/kinetic_{timestamp}.png")
    plt.clf()

    thermal_map = yt.YTArray(np.zeros((1000, 1000, 1))) * yt.units.erg / yt.units.cm**3
    mu = 1.4
    m_H = yt.physical_constants.mass_hydrogen

    temp = obj["flash", "temp"][:,:,center_z]
    n = obj["flash", "dens"][:,:,center_z] / (mu * m_H)
    thermal_map[:, :, 0] = ((3/2) * k * temp * n).to(thermal_map.units)
            
            
    plt.imshow(np.log10(thermal_map), cmap = 'RdBu', vmin = -16, vmax = -9)
    plt.colorbar()
    plt.savefig(f"./analysis/graphs/thermal_{timestamp}.png")
    plt.clf()



    # for i, mask in enumerate(masks):
    #     print(f"reading mask {i}")
    #     mask_img = cv.imread(os.path.join(root_dir, str(timestamp), mask), cv.IMREAD_GRAYSCALE)
    #     mask_boolean = mask_img == 255

    #     z = pixel2pc(int(mask.split(".")[-2].split("z")[-1]), x_y_z="z")  # Assuming pixel2pc is defined elsewhere

    #     temp = obj["flash", "temp"][:, :, z]
    #     n = obj["flash", "dens"][:, :, z] / (mu * m_H)  # Assuming mu and m_H are defined elsewhere

    #     rho = obj["flash", "dens"][:, :, z]
    #     v_sq = obj["flash", "velx"][:, :, z]**2 + obj["flash", "vely"][:, :, z]**2 + obj["flash", "velz"][:, :, z]**2

    #     cell_volume = obj["flash", "cell_volume"][:, :, z]

    #     print(cell_volume.shape)
    #     print(kinetic_energy.shape)

    #     kinetic_energy = (0.5 * rho * v_sq * cell_volume).to('erg')
    #     thermal_energy = ((3/2) * k * temp * n * cell_volume).to('erg')  

        
    #     # timestamp_energy['kinetic_energy'] += np.sum(kinetic_energy[mask_boolean])
    #     # timestamp_energy['thermal_energy'] += np.sum(thermal_energy[mask_boolean])
    #     # timestamp_energy['total_energy'] += np.sum(kinetic_energy[mask_boolean] + thermal_energy[mask_boolean])

    #     # Update kinetic and thermal energy maps
    #     kinetic_energy_map += kinetic_energy  #.to_ndarray()
    #     thermal_energy_map += thermal_energy  #.to_ndarray()

    #     #DEBUG
    #     break

    # # Save both kinetic and thermal energy maps, or use them as needed
    # mask_img = cv.imread(os.path.join(root_dir, str(timestamp), center_mask_path), cv.IMREAD_GRAYSCALE)
    # _, binary_mask = cv.threshold(mask_img, 127, 255, cv.THRESH_BINARY)
    # contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # for contour in contours:
    #     print(contour)

    #     epsilon = 0.01 * cv.arcLength(contour, True)  # 1% of the arc length
    #     approx_polygon = cv.approxPolyDP(contour, epsilon, True)
        
    #     # Step 4: Draw the polygon
    #     cv.drawContours(kinetic_energy_map, [approx_polygon], 0, (255, 255, 255), 1)  # Draw white polygon
    #     cv.drawContours(thermal_energy_map, [approx_polygon], 0, (255, 255, 255), 1)
    
    
    # For example, to save as numpy arrays:
    # plt.imshow(np.log10(kinetic_energy_map), cmap = 'RdBu', vmin = -16, vmax = -9)
    # plt.colorbar()
    # plt.savefig(os.path.join(output_dir, f"kinetic_map_{timestamp}.png"))
    # plt.clf()
    # plt.imshow(np.log10(thermal_energy_map), cmap = 'RdBu', vmin = -16, vmax = -9)
    # plt.colorbar()
    # plt.savefig(os.path.join(output_dir, f"thermal_map_{timestamp}.png"))
    # plt.clf()

    # cv.imwrite(os.path.join(output_dir, f"kinetic_map_{timestamp}.png"), kinetic_energy_map)
    # cv.imwrite(os.path.join(output_dir, f"thermal_map_{timestamp}.png"), thermal_energy_map)
    


    return timestamp_energy




def main(args):
    timestamps = os.listdir(args.mask_root)
    timestamps = [int(timestamp) for timestamp in sorted(timestamps) if os.path.isdir(os.path.join(args.mask_root, timestamp))] 
    energy_data = {}

    for timestamp in timestamps:
        #DEBUG
        print(f"Processing {timestamp}")
        hdf5_filename = os.path.join(args.hdf5_root, f"{args.file_prefix}{timestamp}")
        timestamp_energy = calc_energy(hdf5_filename, args.mask_root, timestamp, args.xlim, args.ylim, args.zlim, args.output_root)
        energy_data[timestamp] = timestamp_energy


    # # Plotting
    # timestamps = list(energy_data.keys())
    # kinetic_energies = [energy_data[timestamp]['kinetic_energy'] for timestamp in timestamps]
    # thermal_energies = [energy_data[timestamp]['thermal_energy'] for timestamp in timestamps]
    # total_energies = [energy_data[timestamp]['total_energy'] for timestamp in timestamps]

    # plot_energy(timestamps, kinetic_energies, thermal_energies, total_energies, args.output_root)

    # # Accumulated total energy
    # print(f"Accumulated Kinetic Energy: {sum(kinetic_energies)} erg")
    # print(f"Accumulated Thermal Energy: {sum(thermal_energies)} erg")
    # print(f"Accumulated Total Energy: {sum(total_energies)} erg")


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









