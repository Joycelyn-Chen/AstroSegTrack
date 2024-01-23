import yt
import os
import pandas as pd
import argparse
from Data.utils import *

def read_dat_log(dat_file_root, dataset_root):
    # Only 1 log file is enough, cause they're actually copies of each other
    # dat_files = glob.glob(os.path.join(dataset_root, dat_file_root, "*.dat"))
    dat_files = [os.path.join(dataset_root, dat_file_root, "SNfeedback.dat")]
    
    # Initialize an empty DataFrame
    all_data = pd.DataFrame()

    # Read and concatenate data from all .dat files
    for dat_file in dat_files:
        # Assuming space-separated values in the .dat files
        df = pd.read_csv(dat_file, delim_whitespace=True, header=None,
                        names=['n_SN', 'type', 'n_timestep', 'n_tracer', 'time',
                                'posx', 'posy', 'posz', 'radius', 'mass'])
        
        # Convert the columns to numerical
        df = df.iloc[1:]
        df['n_SN'] = df['n_SN'].map(int)
        df['type'] = df['type'].map(int)
        df['n_timestep'] = df['n_timestep'].map(int)
        df['n_tracer'] = df['n_tracer'].map(int)
        df['time'] = pd.to_numeric(df['time'],errors='coerce')
        df['posx'] = pd.to_numeric(df['posx'],errors='coerce')
        df['posy'] = pd.to_numeric(df['posy'],errors='coerce')
        df['posz'] = pd.to_numeric(df['posz'],errors='coerce')
        df['radius'] = pd.to_numeric(df['radius'],errors='coerce')
        df['mass'] = pd.to_numeric(df['mass'],errors='coerce')
        all_data = pd.concat([all_data, df], ignore_index=True)
        all_data = all_data.drop(df[df['n_tracer'] != 0].index)

    # Convert time to Megayears
    all_data['time_Myr'] = seconds_to_megayears(all_data['time'])

    # Convert 'pos' from centimeters to parsecs
    all_data['posx_pc'] = cm2pc(all_data['posx'])
    all_data['posy_pc'] = cm2pc(all_data['posy'])
    all_data['posz_pc'] = cm2pc(all_data['posz'])

    # Sort the DataFrame by time in ascending order
    all_data.sort_values(by='time_Myr', inplace=True)

    return all_data


def main(args):
    # Read and process the .dat file logs
    all_data_df = read_dat_log(args.dat_file_root, args.dataset_root)

    # Loop through all time and record the explosion    
    for timestamp in range(args.begin_timestamp, args.end_timestamp, args.incr):
        # DEBUG
        print(f"Processing timestamp {timestamp}")

        time_Myr = timestamp2time_Myr(timestamp)
        filtered_data = filter_data(all_data_df[(all_data_df['time_Myr'] >= time_Myr - 1) & (all_data_df['time_Myr'] <= time_Myr)], (low_x0, low_y0, low_w, low_h, bottom_z, top_z))
        
        # read hdf5 file and do projection plot
        filename = f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{timestamp}"
        ds = yt.load(os.path.join(args.hdf5_root, filename))
        prj = yt.ProjectionPlot(ds, 'z', 'dens', center = [0, 0, 0] * yt.units.pc)
        prj.annotate_timestamp()
        prj.annotate_scale()

        # loop through SN explosions and label them
        for i in range(len(filtered_data)):
            prj.annotate_sphere([filtered_data.iloc[i]["posx_pc"], filtered_data.iloc[i]["posy_pc"]], radius=(10, "pc"), coord_system="plot", text="")

        prj.save(os.path.join(args.output_dir, f'{filename}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin_timestamp", help="The starting timestamp for the dataset", type = int)                 # 200
    parser.add_argument("--end_timestamp", help="The ending timestamp for the dataset", type = int)                     # 360
    parser.add_argument("--incr", help="The timestamp increment interval", type = int)                                  # 10
    parser.add_argument("--hdf5_root", help="Path to HDF5 data root")                                                   # "/home/joy0921/Desktop/Dataset/200_360/finer_time_200_360_original"
    parser.add_argument("--output_dir", help="Output directory for the projection plots")                               # "/home/joy0921/Desktop/Dataset/200_360/ProjectionPlots"
    parser.add_argument("--dataset_root", help="The root directory to the dataset")                                     # "../../Dataset"
    parser.add_argument("--dat_file_root", help="The root directory to the SNfeedback files, relative to dataset root")         # "SNfeedback"

    # python Data/label_explosions.py --begin_timestamp 200 --end_timestamp 360 --incr 10 --hdf5_root "/home/joy0921/Desktop/Dataset/200_360/finer_time_200_360_original" --output_dir "/home/joy0921/Desktop/Dataset/200_360/ProjectionPlots" --dataset_root "../../Dataset" --dat_file_root "SNfeedback" > output.txt 2>&1 &
    
    args = parser.parse_args()
    main(args)