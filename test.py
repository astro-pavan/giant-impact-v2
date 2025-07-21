# type: ignore

import numpy as np
import matplotlib.pyplot as plt
import re

from photosphere import photosphere, M_earth, L_sun, yr

snapshot_path = '/data/pt426/Final_Sims/'

directory = ['',
             'target mass/', 'target mass/', 'target mass/', 'target mass/', 'target mass/', 'target mass/', 'target mass/',
             'mass ratio/', 'mass ratio/', 'mass ratio/',
             'mass ratio/target mass/', 'mass ratio/target mass/', 'mass ratio/target mass/', 'mass ratio/target mass/',
             'impact parameter/', 'impact parameter/', 'impact parameter/', 'impact parameter/', 'impact parameter/', 'impact parameter/', 'impact parameter/',
             'impact parameter/target mass/', 'impact parameter/target mass/', 'impact parameter/target mass/', 'impact parameter/target mass/']

sim_name = ['impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.50_spin0.0',  # 0

            'impact_p1.0e+05_M0.1_ratio1.00_v1.10_b0.50_spin0.0',  # 1 *
            'impact_p1.0e+05_M0.2_ratio1.00_v1.10_b0.50_spin0.0',  # 2
            'impact_p1.0e+05_M0.4_ratio1.00_v1.10_b0.50_spin0.0',  # 3
            'impact_p1.0e+05_M0.8_ratio1.00_v1.10_b0.50_spin0.0',  # 4
            'impact_p1.0e+05_M1.0_ratio1.00_v1.10_b0.50_spin0.0',  # 5
            'impact_p1.0e+05_M1.5_ratio1.00_v1.10_b0.50_spin0.0',  # 6
            'impact_p1.0e+05_M2.0_ratio1.00_v1.10_b0.50_spin0.0',  # 7

            'impact_p1.0e+05_M0.5_ratio0.05_v1.10_b0.50_spin0.0',  # 8
            'impact_p1.0e+05_M0.5_ratio0.20_v1.10_b0.50_spin0.0',  # 9 *
            'impact_p1.0e+05_M0.5_ratio0.50_v1.10_b0.50_spin0.0',  # 10

            'impact_p1.0e+05_M0.1_ratio0.50_v1.10_b0.50_spin0.0',  # 11
            'impact_p1.0e+05_M0.2_ratio0.50_v1.10_b0.50_spin0.0',  # 12
            'impact_p1.0e+05_M1.0_ratio0.50_v1.10_b0.50_spin0.0',  # 13
            'impact_p1.0e+05_M2.0_ratio0.50_v1.10_b0.50_spin0.0',  # 14

            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.10_spin0.0',  # 15
            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.20_spin0.0',  # 16
            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.30_spin0.0',  # 17
            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.40_spin0.0',  # 18
            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.60_spin0.0',  # 19
            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.70_spin0.0',  # 20 *
            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.80_spin0.0',  # 21 *

            'impact_p1.0e+05_M0.1_ratio1.00_v1.10_b0.10_spin0.0',  # 22 *
            'impact_p1.0e+05_M0.2_ratio1.00_v1.10_b0.10_spin0.0',  # 23
            'impact_p1.0e+05_M1.0_ratio1.00_v1.10_b0.10_spin0.0',  # 24
            'impact_p1.0e+05_M2.0_ratio1.00_v1.10_b0.10_spin0.0'   # 25
            ]

snapshot_names = [
    'snapshot_0003.hdf5',  # t = 0.0 hrs
    'snapshot_0006.hdf5',  # t = 0.5 hrs
    'snapshot_0015.hdf5',  # t = 2.0 hrs
    'snapshot_0051.hdf5',  # t = 8.0 hrs
    'snapshot_0240.hdf5'   # t = 40 hrs
]

# LOADS SIMULATION INFO #

n_sims = len(sim_name)
m_target = np.zeros(n_sims)
m_impactor = np.zeros(n_sims)
impact_parameter = np.zeros(n_sims)
v_impact_x = np.zeros(n_sims)
v_impact_y = np.zeros(n_sims)
v_over_v_esc = np.full_like(m_target, 1.1)
n_particles = np.zeros(n_sims)

# extracts simulation properties from text file created by the initial condition generator
for i in range(n_sims):

    with open(f'{snapshot_path}{directory[i]}{sim_name[i]}/{sim_name[i]}.txt', 'r') as file:
        data = file.readlines()

        pattern = r':\s*([\d.]+)\s*'

        m_target[i] = float(re.search(pattern, data[0]).group(1))
        m_impactor[i] = float(re.search(pattern, data[2]).group(1))
        impact_parameter[i] = float(re.search(pattern, data[5]).group(1))
        n_particles[i] = float(re.search(pattern, data[6]).group(1))

        pattern = r':\s*\[([-+\d. ]+)\]'
        matches = re.findall(pattern, data[4])
        numbers = [float(num) for num in matches[0].split()]

        v_impact_x[i] = numbers[0]
        v_impact_y[i] = numbers[1]

snapshot_times = [0, 0.5, 2, 8, 40]  # in hours

v_impact = np.sqrt(v_impact_x ** 2 + v_impact_y ** 2)
# Q_prime = modified_specific_impact_energy(m_target * M_earth, m_impactor* M_earth, v_impact, impact_parameter)

mass_indexes = [2, 3, 0, 4, 5, 6, 7]

impact_parameter_indexes = [15, 16, 17, 18, 0, 19]
mass_ratio_indexes = [8, 9, 10, 0]

mass_mass_ratio_indexes = [11, 12, 10, 14]
mass_impact_parameter_indexes = [23, 15, 24, 25]


# gets the filename of a simulation from the simulation index and time index
def get_filename(i, i_time):
    return f'{snapshot_path}{directory[i]}{sim_name[i]}/{snapshot_names[i_time]}'

p1 = photosphere(get_filename(17, 4))
p1.plot('profile')
t, L, R, T, t_half, t_tenth = p1.cool(5 * yr, n=2000)

plt.plot(t / yr, L / L_sun)
plt.xlabel('t')
plt.ylabel('L')
plt.savefig('cooling.png')