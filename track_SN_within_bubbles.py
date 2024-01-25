import cv2 
import glob
import os
import argparse
import numpy as np
import yt
import json
import matplotlib.pyplot as plt
from Data.utils import *

existence_thres = 0.3

class Tracklet:
    def __init__(self, name, time, center_x, center_y, center_z, mask):
        self.name = name
        self.time = time
        self.center = (center_x, center_y, center_z)
        self.mask = mask
        self.explosions = []

    def add_explosion(self, explosion):
        self.explosions.append(explosion)       #{'time': timestamp2time_Myr(time), 'center': (center_x, center_y, center_z), 'mask': mask, 'volume': volume_pix}

    def remove_mask_from_explosions(self):
        for explosion in self.explosions:
            for evolution in explosion:
                evolution.pop('mask')   

def track_existed(parent, center_z, timestamp, tracks):
    # read the mask
    current_mask = load_mask(parent, timestamp, f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{timestamp}_z{center_z}.png")

    # compute iou with previous tracks
    for prev_tracklet in tracks:
        parent_mask = prev_tracklet.mask
        if compute_iou(current_mask, parent_mask) > existence_thres:
            return prev_tracklet
    return None



def process_tracklets(start_timestamp, end_timestamp, interval, dataset_root):
    tracks = []

    #DEBUG
    print("Begin tracking!")

    for timestamp in range(start_timestamp, end_timestamp, interval):
        # get all cases folder path for this timestamp
        parent_folders = glob.glob(os.path.join(dataset_root, f"SN_{timestamp}*"))
        
        for parent in parent_folders:
            # record info, name, time, xyz, mask
            name = parent.split("/")[-1]
            txt_file = glob.glob(os.path.join(parent, "*.txt"))[0]
            center_x, center_y, center_z = pc2pixel(read_info(txt_file, info_col="posx_pc"), x_y_z="x"), pc2pixel(read_info(txt_file, info_col="posy_pc"), x_y_z="y"), pc2pixel(read_info(txt_file, info_col="posz_pc"), x_y_z="z")
            
            #DEBUG
            print(f"Now processing track {name}")

            current_tracklet = track_existed(parent, center_z, timestamp, tracks)
            if current_tracklet is None:
                mask = load_mask(parent, timestamp, f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{timestamp}_z{center_z}.png")
                
                # add a new tracklet
                current_tracklet = Tracklet(name, timestamp2time_Myr(timestamp), center_x, center_y, center_z, mask)

                #DEBUG
                print("Added a new track!")
                
            # add new explosion
            # loop through masks, incrementing in timestamp, calc volume
            explosion = []
            for time in sorted(list(map(int, list_folders(parent)))):
                img_paths = glob.glob(os.path.join(parent, str(time), "*.png"))
                volume_pix = volume_sum_in_mask(img_paths, parent, time)
                explosion.append({'time': timestamp2time_Myr(time), 'center': (center_x, center_y, center_z), 'mask': mask, 'volume': volume_pix})
            current_tracklet.add_explosion(explosion) 
            
            #DEBUG
            print("Found new explosion!")

            tracks.append(current_tracklet)

    return tracks

def track_analysis(result_tracklets, start_timestamp, end_timestamp, interval, output_root, hdf5_root):
    # x values for the volume chart
    # time_x = range(timestamp2time_Myr(start_timestamp), timestamp2time_Myr(end_timestamp), interval / 10)

    #DEBUG
    print("Begin analysing!")

    for tracklet in result_tracklets:
        time = tracklet.time        # time_Myr
        case_name = tracklet.name
        # center = (pixel2pc(tracklet.center[0], x_y_z="x"), pixel2pc(tracklet.center[1], x_y_z="y"), pixel2pc(tracklet.center[2], x_y_z="z"))

        #DEBUG
        print(f"Now processing track {case_name}")

        # put on projection plot
        filename = f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{time_Myr2timestamp(time)}"
        ds = yt.load(os.path.join(hdf5_root, filename))
        prj = yt.ProjectionPlot(ds, 'z', 'dens', center = [0, 0, 0] * yt.units.pc)
        #prj.annotate_timestamp()
        #prj.annotate_scale()

        #DEBUG
        print(f"Completed loading source file")

        for num, explosion in enumerate(tracklet.explosions):
            explosion_time_x = []
            volume_y = []

            for i, evolvement in enumerate(explosion):
                # [{time, center, mask. volume}, {}]
                if i == 1:      # the initial explosion moment position
                    center = (pixel2pc(evolvement["center"][0], x_y_z="x"), pixel2pc(evolvement["center"][1], x_y_z="y"), pixel2pc(evolvement["center"][2], x_y_z="z"))
                    # projection plot the center
                    prj.annotate_sphere([center[0], center[1]], radius=(10, "pc"), coord_system="plot", text=f"{evolvement['time']}")

                    #DEBUG
                    print(f"Center position: ({center[0]}, {center[1]})")

                explosion_time_x.append(evolvement["time"])
                volume_y.append(evolvement["volume"])
            
            # add a line to volume
            plt.plot(explosion_time_x, volume_y, marker = 'o', linestyle = 'dotted')
          
            #DEBUG
            print(f"processed explosion [{num} / {len(tracklet.explosions)}]")
        
        # save projection plot and volume plot
        prj.annotate_timestamp()
        prj.annotate_scale()
        prj.save(os.path.join(output_root, f'{case_name}_project.png'))

        plt.title(f'Case: {case_name} - Volume change for each explosion',fontsize=10)
        plt.xlabel('Time (Myr)',fontsize=10)
        plt.ylabel('Volume (pixel)',fontsize=10)
        plt.yscale('log')
        plt.grid()
        plt.savefig(os.path.join(output_root, f'{case_name}_volume.png'))
        plt.clf()

        #DEBUG
        print("Plots saved!\n")

   
def save_result_tracklets(result_tracklets, output_root):
    json_filename = "tracklets.json"
    for i, tracklet in enumerate(result_tracklets):
        tracklet.mask = None
        tracklet.remove_mask_from_explosions()
        result_tracklets[i] = tracklet.__dict__

    with open(os.path.join(output_root, json_filename), "w") as json_file:
        json.dump(result_tracklets, json_file, indent=2)

def main(args):
    result_tracklets = process_tracklets(args.start_timestamp, args.end_timestamp, args.interval, args.dataset_root)
    track_analysis(result_tracklets, args.start_timestamp, args.end_timestamp, args.interval, ensure_dir(args.output_root), args.hdf5_root)
    save_result_tracklets(result_tracklets, args.output_root)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_timestamp", help="Specify the starting timestamp", default = 200, type = int)     
    parser.add_argument("--end_timestamp", help="Specify the end timestamp for tracking", default = 330, type = int)
    parser.add_argument("--interval", help="Specify the interval between timestamps", default = 1, type = int)    
    parser.add_argument("--dataset_root", help="Path to dataset root", default = "../../Dataset")   
    parser.add_argument("--output_root", help="Path to output root", default = "../../Dataset/ProjectionPlots")   
    parser.add_argument("--hdf5_root", help="Path to HDF5 root", default = "/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc")   
    
    # python Analysis/track_SN_within_bubbles.py --start_timestamp 200 --end_timestamp 330 --interval 10 --dataset_root "../../Dataset/SN_cases_0122" --output_root "../../Dataset/ProjectionPlots"
    args = parser.parse_args()
    main(args)
