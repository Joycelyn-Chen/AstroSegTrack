import matplotlib.pyplot as plt
import yt
import numpy as np
import os
from astropy import units as u
import cv2 as cv
from utils import *
import argparse

axis_table = {"x": 0, "y" : 1, "z" : 2}
axis_table_target = {"z": 0, "y" : 2, "x" : 1}


def get_velz_dens(obj, x_range, y_range, z_range):
    # read a 3D grid of velz and density array
    #velz = obj["flash", "velz"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('km/s').value        
    dens = obj["flash", "dens"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('g/cm**3').value        
    #temp = obj["flash", "temp"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('K').value 

    # print(f"obj.shape: {obj['flash', 'velz'].shape}")
    # print("x, y, z ranges: ", x_range, y_range, z_range)

    dz = obj['flash', 'dz'][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('cm').value
    mp = yt.physical_constants.mp.value # proton mass

    # calculate the density as column density
    coldens = dens * dz / (1.4 * mp)

    return coldens      #, velz, temp


def read_dataset(args, time_Myr):
    ds = yt.load(os.path.join(args.hdf5_root, '{}{}'.format(args.file_prefix, time_Myr2timestamp(time_Myr))))

    center = [0, 0, 0] * yt.units.pc
    arb_center = ds.arr(center, 'code_length')
    xlim, ylim, zlim = args.pixel_boundary, args.pixel_boundary, args.pixel_boundary
    left_edge = arb_center + ds.quan(-500, 'pc')        
    right_edge = arb_center + ds.quan(500, 'pc')
    obj = ds.arbitrary_grid(left_edge, right_edge, dims=(xlim,ylim,zlim))

    return obj
    
def read_target_density(args, mask_dir, timestamp, dens_cube):
    density_target = np.zeros((args.pixel_boundary, args.pixel_boundary, args.pixel_boundary))
    mask_cube = np.zeros((args.pixel_boundary, args.pixel_boundary, args.pixel_boundary))

    # Iterate through all mask files (0.png to 255.png)
    for SN_id in os.listdir(mask_dir):
        print(f"Processing case: {SN_id} for time: {timestamp}")
        # check if path exists
        if(not os.path.exists(os.path.join(mask_dir, SN_id, str(timestamp)))):
            continue

        for z in range(args.pixel_boundary):
            mask_path = os.path.join(mask_dir, SN_id, str(timestamp), f"{z}.png")
            # Load the mask image and convert to a binary mask (1 for the region of interest, 0 otherwise)
            mask_img = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)  #.convert('L')
            
            if(args.dilate):
                kernel = np.ones((2, 2), np.uint8) 
                mask_img = cv2.dilate(mask_img, kernel, iterations=1)
                kernel = np.ones((3, 3), np.uint8)
                mask_img = cv2.erode(mask_img, kernel, iterations=1)
                mask_img = cv2.erode(mask_img, kernel, iterations=1)
                mask_img = cv2.dilate(mask_img, kernel, iterations=1) 

            mask = mask_img / 255  # Normalize to range [0, 1]
            
            # Apply the mask to the corresponding z-slice of the density array
            # density_target[z] = np.where(mask, dens_cube[:, :, z], 0)       # np.nan
            mask_cube[z] = np.logical_or(mask, mask_cube[z])
            
    

    density_target = np.where(mask_cube, dens_cube, 0)

    # print(f"temp mask save at: {mask_dir}/tmp.png")
    # cv.imwrite(f"{mask_dir}/tmp.png", density_target[135])
    
    return density_target

def save_projection_plot(args, dens_cube, density_target, time_Myr):
    # Project along the z-axis by summing
    projection_cube = np.sum(dens_cube , axis=axis_table[args.view])  # Summing along the z-axis
    projection_target = np.sum(density_target , axis=axis_table_target[args.view])

    # Plot the 2D projection
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(np.log10(projection_cube).T, cmap='viridis', origin='lower')
    
    
    if(args.view == "z"):
        filename = f"{time_Myr}_z"
        if args.draw_contour:
            ax.contour(projection_target.T) #, level = [17e20], color = 'black')
            filename = f"{time_Myr}_zc"
        ax.plot(149,177, 'x:r')
        ax.set_xlabel('X-pix')
        ax.set_ylabel('Y-pix') 
         
        
    elif(args.view == "y"):
        filename = f"{time_Myr}_y"
        if args.draw_contour:
            ax.contour(projection_target) #, level = [17e20], color = 'black')
            filename = f"{time_Myr}_yc"
        ax.plot(149,141, 'x:r')
        ax.set_xlabel('X-pix')
        ax.set_ylabel('Z-pix')
        
        
    elif(args.view == "x"):
        filename = f"{time_Myr}_x"
        if args.draw_contour:
            ax.contour(projection_target) #, level = [17e20], color = 'black')
            filename = f"{time_Myr}_xc"
        ax.plot(177,141, 'x:r')
        ax.set_xlabel('Y-pix')
        ax.set_ylabel('Z-pix')
        

    fig.colorbar(im, label='Column Density (log) ($g/cm^3$)')
    
    
    
    
    ax.set_title('Column Density (Cube)')
    
    # plt.show()

    
    fig.savefig(os.path.join(args.pplot_root, f'{filename}.png'))
    print("Projection plot file save as: ", os.path.join(args.pplot_root, f'{filename}.png'))



def main(args):

    for timestamp in range(args.start_timestamp, args.end_timestamp + args.incr, args.incr):
        time_Myr = timestamp2time_Myr(timestamp)
        # obj = read_dataset(args, time_Myr)
        ds = yt.load(os.path.join(args.hdf5_root, '{}{}'.format(args.file_prefix, time_Myr2timestamp(time_Myr))))

        center = [0, 0, 0] * yt.units.pc
        arb_center = ds.arr(center, 'code_length')
        xlim, ylim, zlim = args.pixel_boundary, args.pixel_boundary, args.pixel_boundary
        left_edge = arb_center + ds.quan(-500, 'pc')        
        right_edge = arb_center + ds.quan(500, 'pc')
        obj = ds.arbitrary_grid(left_edge, right_edge, dims=(xlim,ylim,zlim))
        
        # Load the density array 
        dens_cube = get_velz_dens(obj, (0, args.pixel_boundary), (0, args.pixel_boundary), (0, args.pixel_boundary))
        # Path to masks directory
        # mask_dir = os.path.join(args.mask_root, str(time_Myr2timestamp(time_Myr)))
        
        density_target = read_target_density(args, args.mask_root, str(time_Myr2timestamp(time_Myr)), dens_cube)

        save_projection_plot(args, dens_cube, density_target, time_Myr)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_root", help="The root directory to the dataset")          
    parser.add_argument("--hdf5_root", help="The root directory to the dataset")
    parser.add_argument("--pplot_root", help="Path to output root", default = "../../Dataset/Isolated_case")
    parser.add_argument('-st', '--start_timestamp', help='Input the starting timestamp', type = int)                        # 380
    parser.add_argument('-et', '--end_timestamp', help='Input the ending timestamp', type = int)                            # 570
    parser.add_argument('-i', '--incr', help='The timestamp increment unit', default = 1, type = int)
    parser.add_argument("--file_prefix", help="file prefix", default="sn34_smd132_bx5_pe300_hdf5_plt_cnt_0")
    parser.add_argument('-pixb', '--pixel_boundary', help='Input the pixel resolution', default = 256, type = int)
    parser.add_argument('-d', '--dilate', help='If you wanted to dilate the mask for a little bit and see the edge or not', action="store_true")
    parser.add_argument('-dc', '--draw_contour', help='If you wanted to show contour of the target bubbles or not', action="store_true")
    parser.add_argument('-v', '--view', help='Input the projection view point', default = "z")
    
    
    args = parser.parse_args()
    main(args)

# python analysis/observable_sign_contour.py --mask_root /home/joy0921/Desktop/Dataset/MHD-3DIS/masks/ --hdf5_root "/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc" --pplot_root /home/joy0921/Desktop/Dataset/img_pix256/ProjectionPlots -st 380 -et 570 -i 10 -d -dc
    








