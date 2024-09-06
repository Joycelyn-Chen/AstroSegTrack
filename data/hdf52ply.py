import pandas as pd
import os
import matplotlib.pyplot as plt
import yt
import numpy as np
import argparse
import cv2
import open3d as o3d

DEBUG = False


def apply_otsus_thresholding(image, threshold = 120, height = 100):
    # _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # return threshold

    THESHOLD, binary_image = cv2.threshold(image.astype("uint8"), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, binary_image = cv2.threshold(image.astype("uint8"), 0, 255, round(THESHOLD * (height + 1 / 100)))  
    
    if (DEBUG):
        print(f"temperature threshold: {THESHOLD}")  

    return cv2.bitwise_not(binary_image)

def within_range(min, max, target):
    if min < target and max > target:
        return True
    return False

def SN_center_in_bubble(posx_px, posy_px, x1, y1, w, h):
    if within_range(x1, x1 + w, posx_px) and within_range(y1, y1 + h, posy_px):
        return True
    return False

def normalize4thresholding(arr):
    slice = np.log10(arr)
    return ((slice - np.min(slice)) / (np.max(slice) - np.min(slice))) * 255 

def plot_dens_z(dens, center_z):
    fig, ax = plt.subplots()
    im = ax.imshow(np.log10(dens[:, :, center_z].T[::]), cmap='viridis', aspect='auto')
    fig.colorbar(im, label='density ($g*cm^{-2}$)')
    plt.title('Density ($g*cm^{-2}$)')
    plt.xlabel('X')
    plt.ylabel('Y')
    # fig.savefig(f'../expanding_velocity/{time_Myr}/dzoom_{center_z}.png')
    plt.show()

def get_velz_dens_temp(obj, x_range, y_range, z_range):
    # read a 3D grid of velz and density array
    velz = obj["flash", "velz"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('km/s').value        
    dens = obj["flash", "dens"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('g/cm**3').value        
    temp = obj["flash", "temp"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('K').value 

    print(f"obj.shape: {obj['flash', 'velz'].shape}")
    print("x, y, z ranges: ", x_range, y_range, z_range)
    print(f"velz.shape: {velz.shape}\tdens.shape: {dens.shape}\n\n")      
     

    dz = obj['flash', 'dz'][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('cm').value
    mp = yt.physical_constants.mp.value # proton mass

    # calculate the density as column density
    coldens = dens * dz / (1.4 * mp)

    return velz, coldens, temp


# convert seconds to Megayears
def seconds_to_megayears(seconds):
    return seconds / (1e6 * 365 * 24 * 3600)

def cm2pc(cm):
    return cm * 3.24077929e-19

def timestamp2Myr(timestamp):
    return (timestamp - 200) * 0.1 + 191

def time_Myr2timestamp(time_Myr):
    return round(10 * (time_Myr - 191) + 200)

def max_pooling(coords, grid_size):
    # Quantize coordinates into voxel grid
    quantized_coords = np.floor(coords / grid_size).astype(np.int32)

    # Create a dictionary to store max pooled points
    voxel_dict = {}

    for i in range(len(coords)):
        voxel = tuple(quantized_coords[i])
        if voxel in voxel_dict:
            # Compare current point with the stored one, keep the one with the max value
            voxel_dict[voxel] = np.maximum(voxel_dict[voxel], coords[i])
        else:
            voxel_dict[voxel] = coords[i]

    # Convert the dictionary back to an array
    pooled_coords = np.array(list(voxel_dict.values()))
    
    return pooled_coords


def main(args):

    for time_Myr in range(args.start_time, args. end_time):

        ds = yt.load(os.path.join(args.hdf5_root, '{}0{}'.format(args.file_prefix, time_Myr2timestamp(time_Myr))))

        center = [0, 0, 0] * yt.units.pc
        arb_center = ds.arr(center, 'code_length')
        left_edge = arb_center + ds.quan(-500, 'pc')
        right_edge = arb_center + ds.quan(500, 'pc')
        obj = ds.arbitrary_grid(left_edge, right_edge, dims=(args.xlim, args.ylim, args.zlim))

        x_range_scaled = (0, args.xlim - 1) 
        y_range_scaled = (0, args.ylim - 1)  
        z_range_scaled = (0, args.zlim - 1)

        _, density, temperature = get_velz_dens_temp(obj, x_range_scaled, y_range_scaled, z_range_scaled) 



        # input_point = (center_x, center_y)
        mask = np.zeros((args.pixel_boundary, args.pixel_boundary, args.upper_bound - args.lower_bound))

        # for current_z in range(args.upper_bound - args.lower_bound):
        #     temp_slice = normalize4thresholding(temperature[:, :, current_z + args.lower_bound]) 
            
        #     # threshold + connected component
        #     binary_mask = apply_otsus_thresholding(temp_slice, args.threshold, height=current_z)
        #     _, labels, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        #     # retrieve all mask only
        #     i = 0
        #     binary_mask = labels == i
        #     binary_mask = ~binary_mask
        #     mask[:, :, current_z] = binary_mask

        mask = temperature > args.tempT

        # cast the ring of mask over velz and dens array, plot out the velocity profile, with integrated density as a function of velz  
        dens_roi = np.where(mask, density[:, :, args.lower_bound:args.upper_bound], np.nan)
        coords = np.argwhere(~np.isnan(dens_roi))

        # Determine the grid size to reduce points by approximately 1/40th
        target_size = len(coords) // args.shrink_ratio
        volume = np.ptp(coords, axis=0)  # Get the range of each dimension
        grid_size = np.cbrt(np.prod(volume) / target_size)  # Calculate grid size for target reduction

        # Apply max pooling
        pooled_coords = max_pooling(coords, grid_size)
        densities = np.log10(dens_roi[pooled_coords[:, 0], pooled_coords[:, 1], pooled_coords[:, 2]])
        # Normalize the density values to be between 0 and 1 (as colors are usually in this range)
        densities_normalized = (densities - densities.min()) / (densities.max() - densities.min())


        print(f'Original size: {coords.shape}')
        print(f'Reduced size: {pooled_coords.shape}')

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pooled_coords)
        # Use the 'colors' attribute to store the density values as grayscale colors
        # Repeat the normalized density values across three columns to simulate grayscale (R=G=B)
        # pcd.colors = o3d.utility.Vector3dVector(np.tile(densities_normalized[:, None], (1, 3)))

        o3d.io.write_point_cloud(os.path.join(args.ply_root, f"{time_Myr}.ply"), pcd)
        
        if(DEBUG):
            print(f"Done processing time: {time_Myr}. ply file stored at: {args.ply_root}/{time_Myr}.ply")








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-hr', '--hdf5_root', help='Input the root path to where hdf5 files are stored.', default = "/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc" )       #  "/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc"
    parser.add_argument('-st', '--start_time', help='Input the starting time in Myr', type = int)                        # 206
    parser.add_argument('-et', '--end_time', help='Input the ending time in Myr', type = int)                            # 235                            
    parser.add_argument('-pixb', '--pixel_boundary', help='Input the pixel resolution', default = 255, type = int)
    parser.add_argument('-lb', '--lower_bound', help='The lower bound for the cube.', default = 0, type = int)
    parser.add_argument('-up', '--upper_bound', help='The upper bound for the cube.', default = 255, type = int)
    parser.add_argument("--xlim", help="Input xlim", type = int, default = 256)                                         # 256 
    parser.add_argument("--ylim", help="Input ylim", type = int, default = 256)                                         # 256  
    parser.add_argument("--zlim", help="Input zlim", type = int, default = 256)                                         # 256
    parser.add_argument('-pr', '--ply_root', help='Input the root path to where the k3d plots should be stored')                # '/home/joy0921/Desktop/Dataset/img_pix256/k3d_html'
    parser.add_argument("--file_prefix", help="file prefix", default="sn34_smd132_bx5_pe300_hdf5_plt_cnt_")                                                   # "sn34_smd132_bx5_pe300_hdf5_plt_cnt_"
    parser.add_argument('-t', '--threshold', help='The threshold for slicing density (0-255 scale)', default = 120, type = int)
    parser.add_argument('-sr', '--shrink_ratio', help='The proportion ration for the cube to be shrinked.', default = 20, type = int)
    parser.add_argument('-tempT', '--tempT', help='The temperature threshold for the high temperature, low density region', default = 50000, type = int)
    
    
    # parser.add_argument('-', '--', help='')


    args = parser.parse_args()
    main(args)

    # python hdf52ply.py -st 209 -et 210 --ply_root /home/joy0921/Desktop/Dataset/img_pix256/plys -t 110