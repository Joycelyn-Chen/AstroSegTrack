import matplotlib.pyplot as plt
import astropy
import yt
import numpy as np
import os 

hdf5_root = "/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc"
# hdf5_root = "/home/joy0921/Desktop/Dataset/hdf5_files"

ds = yt.load(os.path.join(hdf5_root, 'sn34_smd132_bx5_pe300_hdf5_plt_cnt_0206'))

center = [0, 0, 0] * yt.units.pc
arb_center = ds.arr(center, 'code_length')
xlim = 256
ylim = 256
zlim= 256
left_edge = arb_center - ds.quan(xlim // 2, 'pc')
right_edge = arb_center + ds.quan(ylim // 2, 'pc')
obj = ds.arbitrary_grid(left_edge, right_edge, dims=[xlim, ylim, zlim])

def get_velx_dens(x, y, z):
    posx_pc = int(x*256/1000)
    posy_pc = int(y*256/1000)
    posz_pc = int(z*256/1000)

    velz = obj["flash", "velz"][posx_pc, posy_pc, -250:250].to('km/s').value        # -250:250
    dens = obj["flash", "dens"][posx_pc, posy_pc, -250:250].to('g/cm**3').value                   # -250:250

    dz = obj['flash', 'dz'][posx_pc, posy_pc, -250:250].to('cm').value
    mp = yt.physical_constants.mp.value # proton mass
    coldens = dens * dz / (1.4 * mp)

    # fig, ax = plt.subplots()
    # ax.plot(velz)
    # axr = ax.twinx()
    # axr.plot(dens, 'k')
    return velz, coldens

velz = []
dens = []
coords = [(-333, -443, -91), (-323, -443, -91), (-333, -453, -91), (-323, -453, -91)]

for coord in coords:
    new_velz, new_dens = get_velx_dens(coord[0], coord[1], coord[2])
    velz.extend(new_velz)
    dens.extend(new_dens)

# plt.plot(velz, dens, '.')
sortidx = np.argsort(velz)
# for v, d in zip(velz[sortidx], dens[sortidx]):
#     print('{:.3f} km/s {:.3e}'.format(v, d))

# the recorded velz
velz = np.array(velz)
dens = np.array(dens)

# velz sorted, density sort with the same order
velz = velz[sortidx]
dens = dens[sortidx]


bins = np.linspace(velz.min(), velz.max(), 100)  # 21 bins means 22 edges
bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Calculate bin centers for plotting

indices = np.digitize(velz, bins)

binned_densities = np.zeros(len(bins)-1)
for i in range(1, len(bins)):
    binned_densities[i-1] = dens[indices == i].sum()

fig, ax = plt.subplots()
ax.bar(bin_centers, binned_densities, width=np.diff(bins), edgecolor='black', align='center')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Density (cm^-2)')
ax.set_title('Line of Sight velosity')
plt.savefig("LoS.png")

plt.clf()

fig, ax = plt.subplots()
ax.plot(velz, label='$v_z$')
axr = ax.twinx()
axr.plot(dens, 'k')
ax.set_xlabel('# data points')
ax.set_ylabel('Velocity (km/s)')
axr.set_ylabel('Column Density (cm^-2)')
ax.set_title('Line of Sight velosity')
ax.legend()

plt.savefig("LoS_dens.png")

fig, ax = plt.subplots()
ax.plot(velz, dens, 'k-')
ax.set_xlabel('$v_z$')
ax.set_ylabel('column density (cm$^{-2}$)')
ax.set_xlim(1, 9)

fig.savefig('dens_velz.png')

idx = np.argmin( np.abs(velz-2.5))
print(idx)
print(dens.shape)

fig, ax = plt.subplots()
im = ax.imshow(np.log10(dens[idx]))

fig.colorbar(im)

fig.savefig('channel_image.png')