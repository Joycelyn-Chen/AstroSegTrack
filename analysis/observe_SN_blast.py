import glob
import os
import argparse
import numpy as np
import yt
import json
import matplotlib.pyplot as plt
from data.utils import *

low_x0, low_y0, low_w, low_h, bottom_z, top_z = -500, -500, 1000, 1000, -500, 500

def main(args):
    # filter data
    all_data = read_dat_log(args.dat_file_root, args.dataset_root)

    for timestamp in range(args.start_timestamp, args.end_timestamp, args.interval):
        time_Myr = timestamp2time_Myr(timestamp)

        #DEBUG
        print(f"Time Myr: {time_Myr}")

        filtered_data = filter_data(all_data[(all_data['time_Myr'] >= time_Myr - args.delta_t) & (all_data['time_Myr'] < time_Myr)],
                                (low_x0, low_y0, low_w, low_h, bottom_z, top_z))

        if len(filtered_data) > 0 :     # there's at least one explosion case
            filename = f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{timestamp}"
            ds = yt.load(os.path.join(args.hdf5_root, filename))
            prj = yt.ProjectionPlot(ds, 'z', 'dens', center = [0, 0, 0] * yt.units.pc)
            prj.annotate_timestamp()
            prj.annotate_scale()
    
            for SN_num in range(len(filtered_data)):
                posx_pc = int(filtered_data.iloc[SN_num]["posx_pc"])
                posy_pc = int(filtered_data.iloc[SN_num]["posy_pc"])
                posz_pc = int(filtered_data.iloc[SN_num]["posz_pc"])
                
                # center_slice_z = pc2pixel(posz_pc, x_y_z = "z")
                time = filtered_data.iloc[SN_num]["time_Myr"]       # time: Myr

                print(f"z: {posz_pc}, time = {time}")   
                

                # projectionplot, column density
                prj.annotate_sphere([posx_pc, posy_pc], radius=(10, "pc"), coord_system="plot", text=f"  {posz_pc} pc")
                # pull all imgs 
                for slice_timestamp in range(timestamp, timestamp + 100, args.interval):        # 100 timestamp = 10 Myr
                    slice_time_Myr = timestamp2time_Myr(slice_timestamp)
                    slice_filename = f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{slice_timestamp}"
                    slice_ds = yt.load(os.path.join(args.hdf5_root, slice_filename))
                    slp = yt.SlicePlot(slice_ds, 'z', 'dens', center = [0, 0, posz_pc] * yt.units.pc)
                    slp.annotate_timestamp()
                    slp.annotate_scale()
                    slice_output_root = ensure_dir(os.path.join(args.output_root, str(timestamp))) 
                    slp.save(os.path.join(slice_output_root, f'{slice_time_Myr}.png'))

            prj.save(os.path.join(args.output_root, str(timestamp), f'{time_Myr}_prj.png'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_timestamp", help="Specify the starting timestamp", default = 200, type = int)     
    parser.add_argument("--end_timestamp", help="Specify the end timestamp for tracking", default = 360, type = int)
    parser.add_argument("--interval", help="Specify the interval between timestamps", default = 1, type = int)
    parser.add_argument("--delta_t", help="Specify the delta t", default = 0.01, type = float)    
    parser.add_argument("--dataset_root", help="Path to dataset root", default = "../../Dataset")   
    parser.add_argument("--dat_file_root", help="The root directory to the SNfeedback files, relative to dataset root")         # "SNfeedback"
    parser.add_argument("--output_root", help="Path to output root", default = "../../Dataset/ProjectionPlots")   
    parser.add_argument("--hdf5_root", help="Path to HDF5 root", default = "/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc")   
    
    # python observe_SN_blast.py --start_timestamp 200 --end_timestamp 360 --interval 1 --delta_t 0.01 --dataset_root "../../Dataset" --dat_file_root "SNfeedback" --output_root "../../Dataset/Explosions" --hdf5_root /home/joy0921/Desktop/Dataset/200_360/finer_time_200_360_original
    args = parser.parse_args()
    main(args)