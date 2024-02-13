import os
import argparse
from utils import ensure_dir

    

if __name__ == "__main__":
    # Replace 'path_to_folder' with the root folder containing your directory structure
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", help="The path to the dataset", default = "/home/joy0921/Desktop/Dataset/SN_cases_0201") 
    parser.add_argument("--case_name", help = "Name of the SN case")        # "SN_2016"
    parser.add_argument("--begin_timestamp", help="timestamp you're doing now", type = int)
    parser.add_argument("--end_timestamp", help="timestamp you're doing now", type = int)
    parser.add_argument("--begin_slice", help="the beginning slice", type = int)
    parser.add_argument("--img_per_batch", help="the end slice to copy", type = int, default = 50)
    parser.add_argument("--destination_root", help="the root of destination", default = "/home/joy0921/Desktop/XMEM/astro/trainval/Annotations")
    parser.add_argument("--shell_filename", help="the name of shell script to store repetitive copying")
    parser.add_argument("--num_movies", help="Total number of movies", type = int)

    args = parser.parse_args()
    
    # with open(args.shell_filename, "w") as f:
    #     for slice_num in range(args.begin_slice, args.begin_slice + args.img_per_batch):
    #         source = os.path.join(args.dataset_root, args.case_name, str(args.timestamp), f"*z{slice_num}*")
    #         output_path = ensure_dir(os.path.join(args.destination_root, str(args.timestamp)))
    #         f.write(f"cp {source} {output_path}\n")

    with open(args.shell_filename, "w") as f:
        for timestamp in range(args.begin_timestamp, args.end_timestamp):
            begin_slice = args.begin_slice
            for movie_num in range(args.num_movies):
                for i in range(args.img_per_batch):
                    slice_num = begin_slice + i
                    source = os.path.join(args.dataset_root, args.case_name, str(timestamp), f"*z{slice_num}*")
                    output_path = ensure_dir(os.path.join(args.destination_root, args.case_name, str(movie_num), str(timestamp)))
                    f.write(f"cp {source} {output_path}\n")

                begin_slice = slice_num + 1
    # python data/reorganize_astro_dataset.py --case_name "SN_21023" --begin_timestamp 210 --end_timestamp 213 --begin_slice 125 --shell_filename "move.sh" --num_movies 10 
