import yt
import os 

# Change the timestamp and the middle z (pixel), will create a clean graph and marks the contour on it (mask size (1000, 1000), img plot size (800, 800))

timestamp = 206
z = 413
image_path = f'./graphs/proj_no_axes_{timestamp}.png'  
mask_path = f'/home/joy0921/Desktop/Dataset/Isloated_case/SN_20617/{timestamp}/sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{timestamp}_z{z}.png'   
output_path = f'./graphs/proj_combined_{timestamp}.png'

center_z = -49

ds = yt.load(os.path.join(hdf5_root, f'sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{timestamp}'))

dens = yt.ProjectionPlot(ds, 'z', 'dens', center=[0, 0, center_z] * yt.units.pc)
dens.hide_colorbar()
dens.hide_axes()
dens.save(image_path)


image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

shrink_factor = 0.8
resized_width = int(mask.shape[1] * shrink_factor)
resized_height = int(mask.shape[0] * shrink_factor)
resized_mask = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

_, thresholded_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresholded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1, (0, 0, 255), 2)  

cv2.imwrite(output_path, image)
