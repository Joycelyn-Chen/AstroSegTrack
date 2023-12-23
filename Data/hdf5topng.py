import yt
import os
# import matplotlib.pyplot as plt
# import numpy as np
# import glob
# from astropy import units as u

root_folder = "/Users/joycelynchen/Desktop/UBC/Research/Data/synthetic_data/sn34"
data_dir = "/Users/joycelynchen/Desktop/UBC/Research/Data/synthetic_data/sn34/img"
file_prefix = "sn34_smd132_bx5_pe300_hdf5_plt_cnt_"

# 0200 ~ 1900
lower_limit = 201
upper_limit = 202
offset = 1
total_slice = 800

for i in range(lower_limit, upper_limit + 1, offset):
    if (i < 1000):
        filename = f"{file_prefix}0{i}"
    else:
        filename = f"{file_prefix}{i}"

    # loading img data
    ds = yt.load(os.path.join(root_folder, filename))
    #ds.current_time, ds.current_time.to('Myr')


    # Saving img
    for j in range(int(-(total_slice / 2)), int(total_slice / 2), 1):
        f = yt.SlicePlot(ds, 'z', 'dens', center = [0, 0, j]*yt.units.pc)
        f.hide_axes()
        f.hide_colorbar()
#         f.show()
        f.save(os.path.join(data_dir, f'{filename}_z{j + (total_slice / 2)}.jpg'))
        
