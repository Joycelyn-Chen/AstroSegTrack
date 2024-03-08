import matplotlib.pyplot as plt
import yt
import numpy as np
import os 

# Initialization and Input 
# hdf5_root = "/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc"
hdf5_root = "/home/joy0921/Desktop/Dataset/hdf5_files"

ds = yt.load(os.path.join(hdf5_root, 'sn34_smd132_bx5_pe300_hdf5_plt_cnt_0206'))

center = [0, 0, 0] * yt.units.pc
arb_center = ds.arr(center, 'code_length')
xlim = 256
ylim = 256
zlim= 256
left_edge = arb_center - ds.quan(xlim // 2, 'pc')
right_edge = arb_center + ds.quan(ylim // 2, 'pc')
obj = ds.arbitrary_grid(left_edge, right_edge, dims=[xlim, ylim, zlim])

def get_velz_dens(x_range, y_range, z_range):
    # read a 3D grid of velz and density array
    velz = obj["flash", "velz"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('km/s').value        
    dens = obj["flash", "dens"][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('g/cm**3').value                   

    dz = obj['flash', 'dz'][x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]].to('cm').value
    mp = yt.physical_constants.mp.value # proton mass

    # calculate the density as column density
    coldens = dens * dz / (1.4 * mp)

    return velz, coldens

velz = []
dens = []


x_range, y_range, z_range = (-358, -308), (-468, -418), (-136, 34)

multiplier = 256 // 1000

# Multiply each element within the tuples by the multiplier
x_range_scaled = tuple(x * multiplier for x in x_range)
y_range_scaled = tuple(y * multiplier for y in y_range)
z_range_scaled = tuple(z * multiplier for z in z_range)

new_velz, new_dens = get_velz_dens(x_range_scaled, y_range_scaled, z_range_scaled)

# convert velz and dens to numpy array
velz = np.array(velz)
dens = np.array(dens)

# Filter the velz and dens arrays
velz_range = (2.5, 2.75)
mask = (velz >= velz_range[0]) & (velz <= velz_range[1])
filtered_dens = dens[mask]

# Channel image
fig, ax = plt.subplots()
im = ax.imshow(np.log10(filtered_dens), cmap='viridis', aspect='auto')
fig.colorbar(im, label='log10(Density)')
plt.title('Filtered Density Channel')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
fig.savefig('channel_image.png')
# plt.show()



