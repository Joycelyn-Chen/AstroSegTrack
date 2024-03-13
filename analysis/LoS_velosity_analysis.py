import os
import yt
import numpy as np
import matplotlib.pyplot as plt
import argparse

def velosity_density_plot(bin_densities, output_root, timestamp, label):
    plt.figure(figsize=(10, 6))
    plt.plot(bins[:-1], bin_densities, drawstyle='steps-post', linewidth=0.5)
    plt.xlim(0, 175)
    # plt.yscale('log')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Accumulated Density (g/cm^3)')
    plt.title('Accumulated Density as a Function of Line of Sight Velocity')
    plt.grid(True)
    plt.savefig(os.path.join(output_root, f"time_{timestamp}_plot_{label}.png"))
    plt.show()


# Initialize parameters

# hdf5_root = "/home/joy0921/Desktop/Dataset/200_360/finer_time_200_360_original"
# hdf5_root = "/home/joy0921/Desktop/Dataset/hdf5_files"
    
hdf5_root = "/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc"
timestamps = [i for i in range(207, 234)]
posx_pc = -333
posy_pc = -443
posz_pc = -91
xlim = 1000
ylim = 1000
zlim = 1000
output_root = "graphs" 
z_range = [(-136, 34), (-250, 250), (-500, 500)]

for timestamp in timestamps:
    # Load the dataset
    filename = f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{timestamp}"
    ds = yt.load(os.path.join(hdf5_root, filename))

    # Set up the grid
    center = [0, 0, 0] * yt.units.pc
    arb_center = ds.arr(center, 'code_length')
    left_edge = arb_center - ds.quan(xlim // 2, 'pc')
    right_edge = arb_center + ds.quan(ylim // 2, 'pc')
    obj = ds.arbitrary_grid(left_edge, right_edge, dims=[xlim, ylim, zlim])

    # 1. Plot the center slice
    dens = yt.SlicePlot(ds, 'z', 'dens', center=[0, 0, posz_pc] * yt.units.pc)
    dens.annotate_marker([posx_pc, posy_pc], 'x', coord_system="plot", plot_args={'color': 'red'})
    dens.save(os.path.join(output_root, f"time_{timestamp}_z{posz_pc}.png"))

    # 2. Read velocity and density data
    data = []
    
    plot_labels = ['whole', 'half', 'all']


    for i, z_ran in enumerate(z_range):
        for z in range(z_ran[0], z_ran[1]):  # Corrected to include 14
            velx = obj["flash", "velx"][posx_pc, posy_pc, z].to('km/s').value
            vely = obj["flash", "vely"][posx_pc, posy_pc, z].to('km/s').value
            velz = obj["flash", "velz"][posx_pc, posy_pc, z].to('km/s').value
            dens = obj["flash", "dens"][posx_pc, posy_pc, z].to('g/cm**3').value
            data.append((velx, vely, velz, dens))

        # 3. Process data and plot
        velocities = [x[2] for x in data] #[np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) for x in data]
        density = [x[3] for x in data]

        # Bin the data
        bins = np.linspace(min(velocities), max(velocities), 400)
        digitized = np.digitize(velocities, bins)
        bin_densities = [np.sum([density[i] for i in range(len(density)) if digitized[i] == j]) for j in range(1, len(bins))]

        # Plot
        velosity_density_plot(bin_densities, output_root, timestamp, plot_labels[i])
        
