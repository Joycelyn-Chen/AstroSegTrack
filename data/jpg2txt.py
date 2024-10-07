import os
import numpy as np
import cv2
import argparse

DEBUG = True

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def read_images(args, timestamp):
    # Create a 3D array (img_arr) for grayscale images (256x256x256)
    img_arr = np.zeros((args.pixel_boundary, args.pixel_boundary, args.pixel_boundary), dtype=np.uint8)
    for z in range(256):
        img_path = os.path.join(args.img_root, str(timestamp), f'{z}.jpg')
        if os.path.exists(img_path):
            img_arr[:, :, z] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img_arr

def threshold_connected(args, img_arr):
    mask_arr = np.zeros((args.pixel_boundary, args.pixel_boundary, args.pixel_boundary), dtype=np.uint8)
    for z in range(args.pixel_boundary):
        image = img_arr[:, :, z]
        THRESHOLD, binary_image = cv2.threshold(image.astype('uint8'), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_image = binary_image / 255
        binary_image = cv2.bitwise_not(binary_image)
        mask_arr[:, :, z] = binary_image
    return mask_arr
    

def read_instance_label(args, timestamp):
    inst_label_arr = np.full((args.pixel_boundary, args.pixel_boundary, args.pixel_boundary), 0, dtype=np.int32)  # Initialize with -100 for background
        
    instance_folders = sorted(os.listdir(args.mask_root))
    
    for label, inst_folder in enumerate(instance_folders, start=1):
        # Each instance folder corresponds to an object and contains 256 slices
        if not os.path.exists(os.path.join(args.mask_root, inst_folder, str(timestamp))):
            continue
        for z in range(args.pixel_boundary):
            mask_path = os.path.join(args.mask_root, inst_folder, str(timestamp), f'{z}.png')
            if os.path.exists(mask_path):
                mask_slice = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                inst_label_arr[:, :, z][mask_slice > 0] = label  # Assign label to non-zero pixels
                if(DEBUG):
                    print(f"labels: {inst_label_arr[:, :, z]}")
    return inst_label_arr

def save2txt(args, timestamp, mask_arr, img_arr, inst_label_arr):
    txt_root = ensure_dir(args.txt_root)
    
    output_file = os.path.join(txt_root, f'{timestamp}_points_GT.txt')  
        
    with open(output_file, 'w') as f:
        for z in range(args.pixel_boundary):
            for y in range(args.pixel_boundary):
                for x in range(args.pixel_boundary):
                    if mask_arr[x, y, z]:  # Check if the point is part of the connected component
                        
                        # Get intensity value from img_arr at (x, y, z) and use it as RGB (r, g, b)
                        intensity = img_arr[x, y, z]
                        r = g = b = intensity
                        
                        # Get instance label from inst_label_arr
                        inst_label = inst_label_arr[x, y, z]

                        # if(DEBUG):
                            # print(f"Instance label: {inst_label}")
                        
                        # Set semantic label based on instance label
                        if inst_label < 50:
                            sem_label = 0  # Category 0
                        else:
                            sem_label = 1  # Category 1
                        
                        # Step 6: Write the point to the output file
                        f.write(f"{x},{y},{z},{r},{g},{b},{sem_label},{inst_label}\n")
    if(DEBUG):
        print(f"Done processing for timestamp: {timestamp}, save as: {output_file}")


def main(args):

    for timestamp in range(args.start_timestamp, args.end_timestamp + 1, args.incr):    # Iterate through all timestamps in the image root folder
        if(DEBUG):
            print(f"Processing time: {timestamp}")
        # Load all images for the current timestamp
        img_arr = read_images(args, timestamp)
        
        if(DEBUG):
            print(f"img shape: {img_arr.shape}\t max: {np.max(img_arr)}")

        # Step 3: Perform thresholding and connected component analysis
        mask_arr = threshold_connected(args, img_arr)
        if(DEBUG):
            print(f"mask shape: {mask_arr.shape}\t max: {np.max(mask_arr)}")
        
        # Step 4: Read the instance masks for the current timestamp
        inst_label_arr = read_instance_label(args, timestamp)
        if(DEBUG):
            print(f"label shape: {inst_label_arr.shape}\t max: {np.max(inst_label_arr)}")

        # Step 5: Generate point cloud data from mask_arr and save to .txt file
        save2txt(args, timestamp, mask_arr, img_arr, inst_label_arr)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_root", help="The root directory to the mask dataset")          
    parser.add_argument("--img_root", help="The root directory to the image dataset")
    parser.add_argument("--txt_root", help="Path to output root", default = ".")
    parser.add_argument('-st', '--start_timestamp', help='Input the starting timestamp', type = int)                        # 380
    parser.add_argument('-et', '--end_timestamp', help='Input the ending timestamp', type = int)                            # 400
    parser.add_argument('-i', '--incr', help='The timestamp increment unit', default = 10, type = int)
    parser.add_argument('-pixb', '--pixel_boundary', help='Input the pixel resolution', default = 256, type = int)

    args = parser.parse_args()

    main(args)

    

# this program read the image stacks as input and output txt file to be converted to point cloud dataset. (stpls3d format)
# python /home/joy0921/Desktop/Dataset/MHD-3DIS/masks