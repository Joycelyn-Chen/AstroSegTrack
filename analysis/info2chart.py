import matplotlib.pyplot as plt
import os

timestamp_info = {
    380: {'volume': 14089, 'kinetic': 1.5005729e+49, 'thermal': 2.65135734e+49, 'total': 4.15193024e+49},
    390: {'volume': 68702, 'kinetic': 5.00933178e+49, 'thermal': 9.21998342e+49, 'total': 1.42293152e+50},
    400: {'volume': 100386, 'kinetic': 9.654345e+49, 'thermal': 1.3981759e+50, 'total': 2.3636104e+50},
    410: {'volume': 149648, 'kinetic': 1.36782795e+50, 'thermal': 2.15469704e+50, 'total': 3.52252499e+50},
    420: {'volume': 189608, 'kinetic': 3.07574381e+50, 'thermal': 9.07682681e+50, 'total': 1.21525706e+51},
    430: {'volume': 266720, 'kinetic': 1.58806869e+50, 'thermal': 2.8964993e+50, 'total': 4.48456799e+50},
    440: {'volume': 255996, 'kinetic': 1.81422658e+50, 'thermal': 3.19503924e+50, 'total': 5.00926582e+50},
    450: {'volume': 236942, 'kinetic': 1.49367085e+50, 'thermal': 3.14695257e+50, 'total': 4.64062343e+50},
    460: {'volume': 219878, 'kinetic': 1.30310779e+50, 'thermal': 2.53593612e+50, 'total': 3.83904392e+50},
    470: {'volume': 196320, 'kinetic': 1.92384296e+50, 'thermal': 5.36514629e+50, 'total': 7.28898925e+50},
    480: {'volume': 218358, 'kinetic': 1.52930685e+50, 'thermal': 4.12116763e+50, 'total': 5.65047448e+50},
    490: {'volume': 211296, 'kinetic': 1.07807331e+50, 'thermal': 2.55761928e+50, 'total': 3.63569259e+50},
    500: {'volume': 219594, 'kinetic': 9.15569463e+49, 'thermal': 2.18312539e+50, 'total': 3.09869485e+50},
    510: {'volume': 267034, 'kinetic': 8.20056499e+49, 'thermal': 2.30496659e+50, 'total': 3.12502309e+50},
    520: {'volume': 91200, 'kinetic': 4.35192219e+49, 'thermal': 1.1268987e+50, 'total': 1.56209092e+50},
    530: {'volume': 69860, 'kinetic': 5.69469626e+49, 'thermal': 1.21583119e+50, 'total': 1.78530082e+50},
    540: {'volume': 92366, 'kinetic': 4.94339317e+49, 'thermal': 1.21644748e+50, 'total': 1.71078679e+50},
    550: {'volume': 230262, 'kinetic': 1.39776873e+50, 'thermal': 3.01649105e+50, 'total': 4.41425978e+50},
    560: {'volume': 166622, 'kinetic': 9.17803683e+49, 'thermal': 2.3107211e+50, 'total': 3.22852478e+50},
    570: {'volume': 2301738, 'kinetic': 1.47610987e+51, 'thermal': 1.87079908e+51, 'total': 3.34690895e+51},
    580: {'volume': 2601688, 'kinetic': 1.330269e+51, 'thermal': 1.84762518e+51, 'total': 3.17789418e+51},
    590: {'volume': 108738, 'kinetic': 7.79758033e+49, 'thermal': 1.56941574e+50, 'total': 2.34917377e+50},
    600: {'volume': 2856838, 'kinetic': 1.80409906e+51, 'thermal': 2.90565567e+51, 'total': 4.70975473e+51}
}

start_timestamp = 380
end_timestamp = 600
interval = 10

output_root = "/Users/joycelynchen/Desktop/UBC/Research/Program/Dataset/SB230"

volume = []
kinetic = []
thermal = []
total = []

def plot_energy(timestamps, kinetic_energies, thermal_energies, total_energies, output_root):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, kinetic_energies, label='Kinetic Energy (erg)')
    plt.plot(timestamps, thermal_energies, label='Thermal Energy (erg)')
    plt.plot(timestamps, total_energies, label='Total Energy (erg)')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Energy (erg)')
    plt.legend()
    plt.tight_layout()
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

plot_volume(timestamps, volume, output_root)
plot_energy(timestamps, kinetic, thermal, total, output_root)
