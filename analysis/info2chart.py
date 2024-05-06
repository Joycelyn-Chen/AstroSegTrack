import matplotlib.pyplot as plt
import os
import yt
from unyt import unyt_quantity
import numpy as np
from utils import *

timestamp_info = {380: {'volume': 14089, 'kinetic': unyt_quantity(1.5005729e+49, 'erg'), 'thermal': unyt_quantity(2.65135734e+49, 'erg'), 'total': unyt_quantity(4.15193024e+49, 'erg'), 'heating': unyt_quantity(1.22276345e-21, 'cm**(-3)'), 'cooling': unyt_quantity(5.14991215e-21, 'erg/(cm**3*s)')}, 390: {'volume': 68702, 'kinetic': unyt_quantity(5.00933178e+49, 'erg'), 'thermal': unyt_quantity(9.21998342e+49, 'erg'), 'total': unyt_quantity(1.42293152e+50, 'erg'), 'heating': unyt_quantity(3.98423043e-21, 'cm**(-3)'), 'cooling': unyt_quantity(1.70266324e-20, 'erg/(cm**3*s)')}, 400: {'volume': 100386, 'kinetic': unyt_quantity(9.654345e+49, 'erg'), 'thermal': unyt_quantity(1.3981759e+50, 'erg'), 'total': unyt_quantity(2.3636104e+50, 'erg'), 'heating': unyt_quantity(5.00460345e-21, 'cm**(-3)'), 'cooling': unyt_quantity(2.14408269e-20, 'erg/(cm**3*s)')}, 410: {'volume': 149648, 'kinetic': unyt_quantity(1.36782795e+50, 'erg'), 'thermal': unyt_quantity(2.15469704e+50, 'erg'), 'total': unyt_quantity(3.52252499e+50, 'erg'), 'heating': unyt_quantity(6.8274708e-21, 'cm**(-3)'), 'cooling': unyt_quantity(3.09066735e-20, 'erg/(cm**3*s)')}, 420: {'volume': 189608, 'kinetic': unyt_quantity(3.07574381e+50, 'erg'), 'thermal': unyt_quantity(9.07682681e+50, 'erg'), 'total': unyt_quantity(1.21525706e+51, 'erg'), 'heating': unyt_quantity(7.96287704e-21, 'cm**(-3)'), 'cooling': unyt_quantity(1.07258598e-19, 'erg/(cm**3*s)')}, 430: {'volume': 266720, 'kinetic': unyt_quantity(1.58806869e+50, 'erg'), 'thermal': unyt_quantity(2.8964993e+50, 'erg'), 'total': unyt_quantity(4.48456799e+50, 'erg'), 'heating': unyt_quantity(8.9215333e-21, 'cm**(-3)'), 'cooling': unyt_quantity(3.91088972e-20, 'erg/(cm**3*s)')}, 440: {'volume': 255996, 'kinetic': unyt_quantity(1.81422658e+50, 'erg'), 'thermal': unyt_quantity(3.19503924e+50, 'erg'), 'total': unyt_quantity(5.00926582e+50, 'erg'), 'heating': unyt_quantity(7.84901105e-21, 'cm**(-3)'), 'cooling': unyt_quantity(3.63295717e-20, 'erg/(cm**3*s)')}, 450: {'volume': 236942, 'kinetic': unyt_quantity(1.49367085e+50, 'erg'), 'thermal': unyt_quantity(3.14695257e+50, 'erg'), 'total': unyt_quantity(4.64062343e+50, 'erg'), 'heating': unyt_quantity(8.8069425e-21, 'cm**(-3)'), 'cooling': unyt_quantity(4.0557089e-20, 'erg/(cm**3*s)')}, 460: {'volume': 219878, 'kinetic': unyt_quantity(1.30310779e+50, 'erg'), 'thermal': unyt_quantity(2.53593612e+50, 'erg'), 'total': unyt_quantity(3.83904392e+50, 'erg'), 'heating': unyt_quantity(8.44521424e-21, 'cm**(-3)'), 'cooling': unyt_quantity(3.62783335e-20, 'erg/(cm**3*s)')}, 470: {'volume': 196320, 'kinetic': unyt_quantity(1.92384296e+50, 'erg'), 'thermal': unyt_quantity(5.36514629e+50, 'erg'), 'total': unyt_quantity(7.28898925e+50, 'erg'), 'heating': unyt_quantity(6.82531839e-21, 'cm**(-3)'), 'cooling': unyt_quantity(6.25701462e-20, 'erg/(cm**3*s)')}, 480: {'volume': 218358, 'kinetic': unyt_quantity(1.52930685e+50, 'erg'), 'thermal': unyt_quantity(4.12116763e+50, 'erg'), 'total': unyt_quantity(5.65047448e+50, 'erg'), 'heating': unyt_quantity(6.57341233e-21, 'cm**(-3)'), 'cooling': unyt_quantity(4.096314e-20, 'erg/(cm**3*s)')}, 490: {'volume': 211296, 'kinetic': unyt_quantity(1.07807331e+50, 'erg'), 'thermal': unyt_quantity(2.55761928e+50, 'erg'), 'total': unyt_quantity(3.63569259e+50, 'erg'), 'heating': unyt_quantity(6.8708245e-21, 'cm**(-3)'), 'cooling': unyt_quantity(3.05709236e-20, 'erg/(cm**3*s)')}, 500: {'volume': 219594, 'kinetic': unyt_quantity(9.15569463e+49, 'erg'), 'thermal': unyt_quantity(2.18312539e+50, 'erg'), 'total': unyt_quantity(3.09869485e+50, 'erg'), 'heating': unyt_quantity(6.73551941e-21, 'cm**(-3)'), 'cooling': unyt_quantity(2.89368665e-20, 'erg/(cm**3*s)')}, 510: {'volume': 267034, 'kinetic': unyt_quantity(8.20056499e+49, 'erg'), 'thermal': unyt_quantity(2.30496659e+50, 'erg'), 'total': unyt_quantity(3.12502309e+50, 'erg'), 'heating': unyt_quantity(7.41501441e-21, 'cm**(-3)'), 'cooling': unyt_quantity(3.07479584e-20, 'erg/(cm**3*s)')}, 520: {'volume': 91200, 'kinetic': unyt_quantity(4.35192219e+49, 'erg'), 'thermal': unyt_quantity(1.1268987e+50, 'erg'), 'total': unyt_quantity(1.56209092e+50, 'erg'), 'heating': unyt_quantity(4.98289354e-21, 'cm**(-3)'), 'cooling': unyt_quantity(1.96787915e-20, 'erg/(cm**3*s)')}, 530: {'volume': 69860, 'kinetic': unyt_quantity(5.69469626e+49, 'erg'), 'thermal': unyt_quantity(1.21583119e+50, 'erg'), 'total': unyt_quantity(1.78530082e+50, 'erg'), 'heating': unyt_quantity(4.28695757e-21, 'cm**(-3)'), 'cooling': unyt_quantity(1.83972807e-20, 'erg/(cm**3*s)')}, 540: {'volume': 92366, 'kinetic': unyt_quantity(4.94339317e+49, 'erg'), 'thermal': unyt_quantity(1.21644748e+50, 'erg'), 'total': unyt_quantity(1.71078679e+50, 'erg'), 'heating': unyt_quantity(5.30462692e-21, 'cm**(-3)'), 'cooling': unyt_quantity(2.03419316e-20, 'erg/(cm**3*s)')}, 550: {'volume': 230262, 'kinetic': unyt_quantity(1.39776873e+50, 'erg'), 'thermal': unyt_quantity(3.01649105e+50, 'erg'), 'total': unyt_quantity(4.41425978e+50, 'erg'), 'heating': unyt_quantity(1.12494031e-20, 'cm**(-3)'), 'cooling': unyt_quantity(4.63097069e-20, 'erg/(cm**3*s)')}, 560: {'volume': 166622, 'kinetic': unyt_quantity(9.17803683e+49, 'erg'), 'thermal': unyt_quantity(2.3107211e+50, 'erg'), 'total': unyt_quantity(3.22852478e+50, 'erg'), 'heating': unyt_quantity(1.16857754e-20, 'cm**(-3)'), 'cooling': unyt_quantity(4.55836795e-20, 'erg/(cm**3*s)')}, 570: {'volume': 2301738, 'kinetic': unyt_quantity(1.47610987e+51, 'erg'), 'thermal': unyt_quantity(1.87079908e+51, 'erg'), 'total': unyt_quantity(3.34690895e+51, 'erg'), 'heating': unyt_quantity(1.83440925e-20, 'cm**(-3)'), 'cooling': unyt_quantity(1.25109239e-19, 'erg/(cm**3*s)')}, 580: {'volume': 2601688, 'kinetic': unyt_quantity(1.330269e+51, 'erg'), 'thermal': unyt_quantity(1.84762518e+51, 'erg'), 'total': unyt_quantity(3.17789418e+51, 'erg'), 'heating': unyt_quantity(2.03977726e-20, 'cm**(-3)'), 'cooling': unyt_quantity(1.18774229e-19, 'erg/(cm**3*s)')}, 590: {'volume': 108738, 'kinetic': unyt_quantity(7.79758033e+49, 'erg'), 'thermal': unyt_quantity(1.56941574e+50, 'erg'), 'total': unyt_quantity(2.34917377e+50, 'erg'), 'heating': unyt_quantity(8.27641503e-21, 'cm**(-3)'), 'cooling': unyt_quantity(3.26759447e-20, 'erg/(cm**3*s)')}, 600: {'volume': 2856838, 'kinetic': unyt_quantity(1.80409906e+51, 'erg'), 'thermal': unyt_quantity(2.90565567e+51, 'erg'), 'total': unyt_quantity(4.70975473e+51, 'erg'), 'heating': unyt_quantity(1.63972572e-20, 'cm**(-3)'), 'cooling': unyt_quantity(1.15507235e-19, 'erg/(cm**3*s)')}}

start_timestamp = 380
end_timestamp = 600
interval = 10
pc2cm = 3.086e18 # 1 pc in cm
Myr2sec = 3.1536e13

output_root = "/home/joy0921/Desktop/Dataset/SB230"

volume = []
kinetic = []
thermal = []
total = []
heating = []
cooling = []

# def plot_energy(timestamps, kinetic_energies, thermal_energies, total_energies, output_root):
#     plt.figure(figsize=(10, 6))
#     plt.plot(timestamps, kinetic_energies, label='Kinetic Energy (erg)')
#     plt.plot(timestamps, thermal_energies, label='Thermal Energy (erg)')
#     plt.plot(timestamps, total_energies, label='Total Energy (erg)')
#     plt.yscale('log')
#     plt.xlabel('Time')
#     plt.ylabel('Energy (erg)')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_root, 'energy_chart.png'))
#     plt.show()

def energy_integral(energy, volume):
    # volume in pixel
    return energy * volume * (3.9*pc2cm) ** 3 * Myr2sec

def plot_energy(timestamps, kinetic_energies, thermal_energies, total_energies, heating, cooling, E_sn, output_root):
    fig, ax1 = plt.subplots()
    ax1.plot(timestamps, kinetic_energies, label='Kinetic Energy (erg)', color = 'wheat')    #
    ax1.plot(timestamps, thermal_energies, label='Thermal Energy (erg)', color = 'xkcd:beige')    # 
    ax1.plot(timestamps, total_energies, label='Total Energy (erg)', color = 'xkcd:tan')        # gold
    ax1.plot(timestamps, E_sn, 'r+')
    ax1.set_yscale('log')
    ax1.set_xlabel('Time (Myr)')
    ax1.set_ylabel('Energy (erg)')
    
    
    
    ax2 = ax1.twinx()
    ax2.plot(timestamps, energy_integral(heating, volume), label = 'Heating (erg)', color = 'lightblue')   #red, tomato
    ax2.plot(timestamps, energy_integral(cooling, volume), label = 'Cooling (erg)', color = 'xkcd:azure')  #powerblue
    # ax2.plot(timestamps, E_sn, label = "E_sn", color = "red")
    ax2.plot(timestamps, energy_integral(heating - cooling, volume) + E_sn, label = "H - C + E_sn", color = "xkcd:blue")
    ax2.set_yscale('log')
    ax2.set_ylabel('heating and cooling (erg*s)')
    
    # fig.tight_layout()
    fig.legend()
    plt.savefig(os.path.join(output_root, 'energy_chart.png'))
    plt.show()

def plot_volume(timestamps, volume, output_root):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, np.multiply(volume, (3.9**3)), 'bo-')
    plt.xlabel('Time (Myr)')
    plt.ylabel('Accumulated Volume (pc^3)')
    plt.title('Volume Evolution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, 'volume.png'))
    plt.show()

timestamps = list(range(start_timestamp, end_timestamp + 1, interval))

for timestamp in timestamps:
    volume.append(timestamp_info[timestamp]['volume'])
    kinetic.append(timestamp_info[timestamp]['kinetic'])
    thermal.append(timestamp_info[timestamp]['thermal'])
    
    
    total.append(timestamp_info[timestamp]['total'])
    heating.append(timestamp_info[timestamp]['heating'])
    heating_values = np.array([h.value for h in heating])
    
    cooling.append(timestamp_info[timestamp]['cooling'])
    cooling_values = np.array([c.value for c in cooling])


# convert timstamps to time_Myr
time_Myr = [timestamp2time_Myr(timestamp) for timestamp in timestamps]  # 209 ~ 231

# supernovae energy injection
one_hot_explosion = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
E_sn = [num * 1e51 for num in one_hot_explosion]
# E_sn = [ one_hot_explosion[i - 1] + injection for i, injection in enumerate(one_hot_explosion)]
# E_sn.pop(0)


# plot_volume(time_Myr, volume, output_root)
plot_energy(time_Myr, kinetic, thermal, total, heating_values, cooling_values, E_sn, output_root)

