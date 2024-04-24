import matplotlib.pyplot as plt
import os
import yt
from unyt import unyt_quantity

timestamp_info = {
    380: {'volume': 14089, 'kinetic': unyt_quantity(1.5005729e+49, 'erg'), 'thermal': unyt_quantity(2.65135734e+49, 'erg'), 'total': unyt_quantity(4.15193024e+49, 'erg'), 'heating': unyt_quantity(1.83314107e-21, 'cm**(-3)'), 'cooling': unyt_quantity(5.14991215e-21, 'cm**(-6)')}, 
    390: {'volume': 68702, 'kinetic': unyt_quantity(5.00933178e+49, 'erg'), 'thermal': unyt_quantity(9.21998342e+49, 'erg'), 'total': unyt_quantity(1.42293152e+50, 'erg'), 'heating': unyt_quantity(2.93623874e-21, 'cm**(-3)'), 'cooling': unyt_quantity(1.70266324e-20, 'cm**(-6)')}, 
    400: {'volume': 100386, 'kinetic': unyt_quantity(9.654345e+49, 'erg'), 'thermal': unyt_quantity(1.3981759e+50, 'erg'), 'total': unyt_quantity(2.3636104e+50, 'erg'), 'heating': unyt_quantity(2.81168346e-21, 'cm**(-3)'), 'cooling': unyt_quantity(2.14408269e-20, 'cm**(-6)')}, 
    410: {'volume': 149648, 'kinetic': unyt_quantity(1.36782795e+50, 'erg'), 'thermal': unyt_quantity(2.15469704e+50, 'erg'), 'total': unyt_quantity(3.52252499e+50, 'erg'), 'heating': unyt_quantity(2.83988508e-21, 'cm**(-3)'), 'cooling': unyt_quantity(3.09066735e-20, 'cm**(-6)')}, 
    420: {'volume': 189608, 'kinetic': unyt_quantity(3.07574381e+50, 'erg'), 'thermal': unyt_quantity(9.07682681e+50, 'erg'), 'total': unyt_quantity(1.21525706e+51, 'erg'), 'heating': unyt_quantity(2.96155149e-21, 'cm**(-3)'), 'cooling': unyt_quantity(1.07258598e-19, 'cm**(-6)')}, 
    430: {'volume': 266720, 'kinetic': unyt_quantity(1.58806869e+50, 'erg'), 'thermal': unyt_quantity(2.8964993e+50, 'erg'), 'total': unyt_quantity(4.48456799e+50, 'erg'), 'heating': unyt_quantity(3.25691832e-21, 'cm**(-3)'), 'cooling': unyt_quantity(3.91088972e-20, 'cm**(-6)')}, 
    440: {'volume': 255996, 'kinetic': unyt_quantity(1.81422658e+50, 'erg'), 'thermal': unyt_quantity(3.19503924e+50, 'erg'), 'total': unyt_quantity(5.00926582e+50, 'erg'), 'heating': unyt_quantity(3.4722895e-21, 'cm**(-3)'), 'cooling': unyt_quantity(3.63295717e-20, 'cm**(-6)')}, 
    450: {'volume': 236942, 'kinetic': unyt_quantity(1.49367085e+50, 'erg'), 'thermal': unyt_quantity(3.14695257e+50, 'erg'), 'total': unyt_quantity(4.64062343e+50, 'erg'), 'heating': unyt_quantity(3.31478948e-21, 'cm**(-3)'), 'cooling': unyt_quantity(4.0557089e-20, 'cm**(-6)')}, 
    460: {'volume': 219878, 'kinetic': unyt_quantity(1.30310779e+50, 'erg'), 'thermal': unyt_quantity(2.53593612e+50, 'erg'), 'total': unyt_quantity(3.83904392e+50, 'erg'), 'heating': unyt_quantity(3.25615788e-21, 'cm**(-3)'), 'cooling': unyt_quantity(3.62783335e-20, 'cm**(-6)')}, 
    470: {'volume': 196320, 'kinetic': unyt_quantity(1.92384296e+50, 'erg'), 'thermal': unyt_quantity(5.36514629e+50, 'erg'), 'total': unyt_quantity(7.28898925e+50, 'erg'), 'heating': unyt_quantity(3.43462998e-21, 'cm**(-3)'), 'cooling': unyt_quantity(6.25701462e-20, 'cm**(-6)')}, 
    480: {'volume': 218358, 'kinetic': unyt_quantity(1.52930685e+50, 'erg'), 'thermal': unyt_quantity(4.12116763e+50, 'erg'), 'total': unyt_quantity(5.65047448e+50, 'erg'), 'heating': unyt_quantity(3.57080748e-21, 'cm**(-3)'), 'cooling': unyt_quantity(4.096314e-20, 'cm**(-6)')}, 
    490: {'volume': 211296, 'kinetic': unyt_quantity(1.07807331e+50, 'erg'), 'thermal': unyt_quantity(2.55761928e+50, 'erg'), 'total': unyt_quantity(3.63569259e+50, 'erg'), 'heating': unyt_quantity(3.30841143e-21, 'cm**(-3)'), 'cooling': unyt_quantity(3.05709236e-20, 'cm**(-6)')}, 
    500: {'volume': 219594, 'kinetic': unyt_quantity(9.15569463e+49, 'erg'), 'thermal': unyt_quantity(2.18312539e+50, 'erg'), 'total': unyt_quantity(3.09869485e+50, 'erg'), 'heating': unyt_quantity(3.22483039e-21, 'cm**(-3)'), 'cooling': unyt_quantity(2.89368665e-20, 'cm**(-6)')},
    510: {'volume': 267034, 'kinetic': unyt_quantity(8.20056499e+49, 'erg'), 'thermal': unyt_quantity(2.30496659e+50, 'erg'), 'total': unyt_quantity(3.12502309e+50, 'erg'), 'heating': unyt_quantity(3.39101616e-21, 'cm**(-3)'), 'cooling': unyt_quantity(3.07479584e-20, 'cm**(-6)')}, 
    520: {'volume': 91200, 'kinetic': unyt_quantity(4.35192219e+49, 'erg'), 'thermal': unyt_quantity(1.1268987e+50, 'erg'), 'total': unyt_quantity(1.56209092e+50, 'erg'), 'heating': unyt_quantity(2.41210673e-21, 'cm**(-3)'), 'cooling': unyt_quantity(1.96787915e-20, 'cm**(-6)')}, 
    530: {'volume': 69860, 'kinetic': unyt_quantity(5.69469626e+49, 'erg'), 'thermal': unyt_quantity(1.21583119e+50, 'erg'), 'total': unyt_quantity(1.78530082e+50, 'erg'), 'heating': unyt_quantity(2.40872019e-21, 'cm**(-3)'), 'cooling': unyt_quantity(1.83972807e-20, 'cm**(-6)')},
    540: {'volume': 92366, 'kinetic': unyt_quantity(4.94339317e+49, 'erg'), 'thermal': unyt_quantity(1.21644748e+50, 'erg'), 'total': unyt_quantity(1.71078679e+50, 'erg'), 'heating': unyt_quantity(2.45967047e-21, 'cm**(-3)'), 'cooling': unyt_quantity(2.03419316e-20, 'cm**(-6)')},
    550: {'volume': 230262, 'kinetic': unyt_quantity(1.39776873e+50, 'erg'), 'thermal': unyt_quantity(3.01649105e+50, 'erg'), 'total': unyt_quantity(4.41425978e+50, 'erg'), 'heating': unyt_quantity(2.80104603e-21, 'cm**(-3)'), 'cooling': unyt_quantity(4.63097069e-20, 'cm**(-6)')}, 
    560: {'volume': 166622, 'kinetic': unyt_quantity(9.17803683e+49, 'erg'), 'thermal': unyt_quantity(2.3107211e+50, 'erg'), 'total': unyt_quantity(3.22852478e+50, 'erg'), 'heating': unyt_quantity(2.64142213e-21, 'cm**(-3)'), 'cooling': unyt_quantity(4.55836795e-20, 'cm**(-6)')}, 
    570: {'volume': 2301738, 'kinetic': unyt_quantity(1.47610987e+51, 'erg'), 'thermal': unyt_quantity(1.87079908e+51, 'erg'), 'total': unyt_quantity(3.34690895e+51, 'erg'), 'heating': unyt_quantity(4.08162766e-21, 'cm**(-3)'), 'cooling': unyt_quantity(1.25109239e-19, 'cm**(-6)')},
    580: {'volume': 2601688, 'kinetic': unyt_quantity(1.330269e+51, 'erg'), 'thermal': unyt_quantity(1.84762518e+51, 'erg'), 'total': unyt_quantity(3.17789418e+51, 'erg'), 'heating': unyt_quantity(4.0730183e-21, 'cm**(-3)'), 'cooling': unyt_quantity(1.18774229e-19, 'cm**(-6)')}, 
    590: {'volume': 108738, 'kinetic': unyt_quantity(7.79758033e+49, 'erg'), 'thermal': unyt_quantity(1.56941574e+50, 'erg'), 'total': unyt_quantity(2.34917377e+50, 'erg'), 'heating': unyt_quantity(2.56412899e-21, 'cm**(-3)'), 'cooling': unyt_quantity(3.26759447e-20, 'cm**(-6)')}, 
    600: {'volume': 2856838, 'kinetic': unyt_quantity(1.80409906e+51, 'erg'), 'thermal': unyt_quantity(2.90565567e+51, 'erg'), 'total': unyt_quantity(4.70975473e+51, 'erg'), 'heating': unyt_quantity(3.91487867e-21, 'cm**(-3)'), 'cooling': unyt_quantity(1.15507235e-19, 'cm**(-6)')}
    }

start_timestamp = 380
end_timestamp = 600
interval = 10

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

def plot_energy(timestamps, kinetic_energies, thermal_energies, total_energies, heating, cooling, output_root):
    fig, ax1 = plt.subplots()
    ax1.plot(timestamps, kinetic_energies, label='Kinetic Energy (erg)', color = 'olivedrab')
    ax1.plot(timestamps, thermal_energies, label='Thermal Energy (erg)', color = 'orange')
    ax1.plot(timestamps, total_energies, label='Total Energy (erg)', color = 'gold')
    ax1.set_yscale('log')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Energy (erg)')
    
    
    ax2 = ax1.twinx()
    ax2.plot(timestamps, heating, label = 'Heating Rate (cm^-3)', color = 'tomato')
    ax2.plot(timestamps, cooling, label = 'Cooling Rate (cm^-6)', color = 'powderblue')
    ax2.set_yscale('log')
    ax2.set_ylabel('heating rate (cm^-3) and cooling (cm^-6)')
    
    # fig.tight_layout()
    fig.legend()
    plt.savefig(os.path.join(output_root, 'energy_chart.png'))
    plt.show()

def plot_volume(timestamps, volume, output_root):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, volume, 'bo-')
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
    cooling.append(timestamp_info[timestamp]['cooling'])


plot_volume(timestamps, volume, output_root)
plot_energy(timestamps, kinetic, thermal, total, heating, cooling, output_root)
