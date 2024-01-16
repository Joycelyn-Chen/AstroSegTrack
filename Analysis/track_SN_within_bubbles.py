import cv2 
import glob
import os
import argparse


# my dataset contains the segmentation masks for each and blob within the astronomy image. under the root directory are the different supernova explosion (SN) cases, named SN_2001, SN_2001, and so on. Then under each SN case folder are the timestamps ranging from the time of current SN case explosion to timestamp 330. A text file is also located under the SN case folder, indicating the start time of the explosion and the center coordinate of the explosion. Under each timestamp folder are all the segmentation masks for the blob that SN explosion locate in. Masks are represented in 2d slices, with one slice per parsec in z axis. So these blobs are actually 3D blobs in 3D data cube. 

# That being said, write a program traversing backward in time, and record those cases that explodes within the previous existing blob. The naming rule for the SN case is I've concatenate the time when it goes off and the id so SN_2001, would be an SN explosion at time 200 with id = 1, and SN_31938 would be explosion at time 319 with id = 38. Traverse backwards, so start with time 329 and work the way back, search for all text record under SN_329*, record the start time, explosion center position x, y, z of each explosion, then read the mask for center slice, so find the image that has the same slice number with the center z coordinate, and read it into a numpy array using cv2.imread(). Store the above tracklets in a class where it contains the name of the track (named with the SN case folder name), the associated next explosion (record the time and center position), and all the center slices mask of the explosion. 

# Then move on to the previous timestamp, i.e. SN_328*. See of there's an explosion goes off within the previous bubble by reading and comparing the center coordinate of the explosion, if the center coordinate is within the mask, then associate this explosion case by adding the time, the center coordinate of explosion and the mask for the center slice to the tracklet. Repeat for all cases and until the starting timestamp 200.  

# For all SN cases for each timestamp, explore the SN cases, add a new tracklet with the current blob haven't been seen before, see if it's a new blob by comparing the iou between the center slice and all the previous center slices. the blob exists if the iou > 0.6.


def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--", help="", type = int) 
    parser.add_argument("--", help="")        

    
    
    args = parser.parse_args()
    main(args)