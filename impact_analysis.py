# main analysis code

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from unyt import Rearth
import re

from snapshot_analysis import snapshot, gas_slice, data_labels
from photosphere import photosphere, M_earth, L_sun, yr
import EOS as fst

day = 3600 * 24

hill_radius = 60
n_theta = 20
n_phi = 20
resolution = 400


def modified_specific_impact_energy(m_tar, m_imp, v_imp, b):
    m_reduced = (m_imp * m_tar) / (m_imp + m_tar)
    Q = (m_reduced * v_imp ** 2) / (2 * (m_imp + m_tar))
    return Q * (1 + m_imp / m_tar) * (1 - b)


early_runs = [
    '/snapshots/low_mass_twin/snapshot_0274.hdf5',
    '/snapshots/basic_twin/snapshot_0411.hdf5',
    '/snapshots/high_mass_twin/snapshot_0360.hdf5'
    '/snapshots/basic_spin/snapshot_0247.hdf5',
    '/snapshots/advanced_spin/snapshot_0316.hdf5'
]

snapshot_path = '/home/pavan/Project/Final_Sims/'

directory = ['',
             'target mass/', 'target mass/', 'target mass/', 'target mass/', 'target mass/', 'target mass/', 'target mass/',
             'mass ratio/', 'mass ratio/', 'mass ratio/',
             'mass ratio/target mass/', 'mass ratio/target mass/', 'mass ratio/target mass/', 'mass ratio/target mass/',
             'impact parameter/', 'impact parameter/', 'impact parameter/', 'impact parameter/', 'impact parameter/', 'impact parameter/', 'impact parameter/',
             'impact parameter/target mass/', 'impact parameter/target mass/', 'impact parameter/target mass/', 'impact parameter/target mass/']

sim_name = ['impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.50_spin0.0',  # 0

            'impact_p1.0e+05_M0.1_ratio1.00_v1.10_b0.50_spin0.0',  # 1 (weird)
            'impact_p1.0e+05_M0.2_ratio1.00_v1.10_b0.50_spin0.0',  # 2
            'impact_p1.0e+05_M0.4_ratio1.00_v1.10_b0.50_spin0.0',  # 3
            'impact_p1.0e+05_M0.8_ratio1.00_v1.10_b0.50_spin0.0',  # 4
            'impact_p1.0e+05_M1.0_ratio1.00_v1.10_b0.50_spin0.0',  # 5
            'impact_p1.0e+05_M1.5_ratio1.00_v1.10_b0.50_spin0.0',  # 6
            'impact_p1.0e+05_M2.0_ratio1.00_v1.10_b0.50_spin0.0',  # 7

            'impact_p1.0e+05_M0.5_ratio0.05_v1.10_b0.50_spin0.0',  # 8
            'impact_p1.0e+05_M0.5_ratio0.20_v1.10_b0.50_spin0.0',  # 9 (weird)
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
            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.70_spin0.0',  # 20 (doesn't work)
            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.80_spin0.0',  # 21 (doesn't work)

            'impact_p1.0e+05_M0.1_ratio1.00_v1.10_b0.10_spin0.0',  # 22 (weird)
            'impact_p1.0e+05_M0.2_ratio1.00_v1.10_b0.10_spin0.0',  # 23
            'impact_p1.0e+05_M1.0_ratio1.00_v1.10_b0.10_spin0.0',  # 24
            'impact_p1.0e+05_M2.0_ratio1.00_v1.10_b0.10_spin0.0'   # 25
            ]

snapshot_names = [
    'snapshot_0003.hdf5',
    'snapshot_0006.hdf5',
    'snapshot_0015.hdf5',
    'snapshot_0051.hdf5',
    'snapshot_0240.hdf5'
]

# load simulation info

n_sims = len(sim_name)
m_target = np.zeros(n_sims)
m_impactor = np.zeros(n_sims)
impact_parameter = np.zeros(n_sims)
v_impact_x = np.zeros(n_sims)
v_impact_y = np.zeros(n_sims)
v_over_v_esc = np.full_like(m_target, 1.1)

for i in range(n_sims):

    with open(f'{snapshot_path}{directory[i]}{sim_name[i]}/{sim_name[i]}.txt', 'r') as file:
        data = file.readlines()

        pattern = r':\s*([\d.]+)\s*'
        m_target[i] = float(re.search(pattern, data[0]).group(1))
        m_impactor[i] = float(re.search(pattern, data[2]).group(1))
        impact_parameter[i] = float(re.search(pattern, data[5]).group(1))

        pattern = r':\s*\[([-+\d. ]+)\]'
        matches = re.findall(pattern, data[4])
        numbers = [float(num) for num in matches[0].split()]

        v_impact_x[i] = numbers[0]
        v_impact_y[i] = numbers[1]

snapshot_times = [0, 0.5, 2, 8, 40]  # in hours

m_target = m_target * M_earth
m_impactor = m_impactor * M_earth

v_impact = np.sqrt(v_impact_x ** 2 + v_impact_y ** 2)
Q_prime = modified_specific_impact_energy(m_target, m_impactor, v_impact, impact_parameter)


mass_indexes = [2, 3, 0, 4, 5, 6, 7]

impact_parameter_indexes = [15, 16, 17, 18, 0, 19]
mass_ratio_indexes = [8, 9, 10, 0]

mass_mass_ratio_indexes = [11, 12, 10, 14]  # 13 removed
mass_impact_parameter_indexes = [23, 15, 24, 25]


def get_filename(i, i_time):
    return f'{snapshot_path}{directory[i]}{sim_name[i]}/{snapshot_names[i_time]}'


def light_curves(indexes):

    Q = Q_prime[indexes]
    L0 = []
    AM = []
    SAM = []

    light_curve = []
    time = []
    t_half = []
    t_quarter = []
    t_tenth = []

    for i in indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0.append(phot.luminosity / L_sun)
        t, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2(save_name=f'impact{i}', plot=False,
                                                                          plot_interval=0.1)
        light_curve.append(lum)
        time.append(t)
        AM.append(snap.total_angular_momentum)
        SAM.append(snap.total_specific_angular_momentum)
        t_half.append(t2)
        t_quarter.append(t4)
        t_tenth.append(t10)

    t_half = np.array(t_half)
    t_quarter = np.array(t_quarter)
    t_tenth = np.array(t_tenth)

    plt.figure(figsize=(8, 6))

    change_mass_indexes = [1, 2, 0, 3, 4]
    for i in change_mass_indexes:
        plt.plot(time[i] / yr, light_curve[i] / L_sun,
                 label=f'Total mass = {(m_target[i] + m_impactor[i]) / M_earth:.2f}' + '$M_{\oplus}$')
    plt.xlabel('Time (days)')
    plt.ylabel('Luminosity ($L_{\odot}$)')
    plt.legend()

    plt.xlim([0, 40])
    plt.savefig('figures/mass_light_curve_very_long.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_light_curve_very_long.png', bbox_inches='tight')

    plt.xlim([0, 5])
    plt.savefig('figures/mass_light_curve_long.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_light_curve_long.png', bbox_inches='tight')

    plt.xlim([0, 1])
    plt.savefig('figures/mass_light_curve_short.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_light_curve_short.png', bbox_inches='tight')

    plt.xlim([0.05, 20])
    plt.xscale('log')
    plt.savefig('figures/mass_light_curve_log.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_light_curve_log.png', bbox_inches='tight')

    plt.close()

    total_mass = (m_target + m_impactor)[indexes]
    plt.scatter(total_mass[change_mass_indexes] / M_earth, t_half[change_mass_indexes] / day, label='$t_{1/2}$')
    # plt.scatter(total_mass[change_mass_indexes] / M_earth, t_quarter[change_mass_indexes] / yr, label='$t_{1/4}$')
    # plt.scatter(total_mass[change_mass_indexes] / M_earth, t_tenth[change_mass_indexes] / yr, label='$t_{1/10}$')
    plt.xlabel('Total mass ($M_{\oplus}$)')
    plt.ylabel('Cooling time (days)')

    plt.savefig('figures/mass_timescales.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_timescales.png', bbox_inches='tight')
    plt.close()

    plt.scatter(Q, L0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Modified specific Impact Energy (J/kg)')
    plt.ylabel('Luminosity ($L_{\odot}$)')

    plt.scatter(total_mass, L0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Modified specific Impact Energy (J/kg)')
    plt.ylabel('Luminosity ($L_{\odot}$)')

    plt.savefig('figures/QL_plot.png', bbox_inches='tight')
    plt.savefig('figures/QL_plot.pdf', bbox_inches='tight')
    plt.close()

    for i in indexes:
        mass_ratio = m_impactor / m_target
        print(f'{m_target[i] / M_earth:.1f} &'
              f' {impact_parameter[i]:.1f} &'
              f' {mass_ratio[i]:.2f} &'
              f' {v_over_v_esc[i]:.1f} &'
              f' {L0[i] / L_sun:.1e} &'
              f' {t_half[i] / yr:.2f}  \\\\')


def changing_mass():
    Q = Q_prime[mass_indexes]
    L0, AM, SAM = [], [], []

    light_curve, time = [], []
    t_half, t_quarter, t_tenth = [], [], []

    for i in mass_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0.append(phot.luminosity / L_sun)
        t, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2(save_name=f'impact{i}', plot=False,
                                                                          plot_interval=0.1)
        light_curve.append(lum)
        time.append(t)
        AM.append(snap.total_angular_momentum)
        SAM.append(snap.total_specific_angular_momentum)
        t_half.append(t2)
        t_quarter.append(t4)
        t_tenth.append(t10)

    t_half, t_quarter, t_tenth = np.array(t_half), np.array(t_quarter), np.array(t_tenth)

    plt.figure(figsize=(8, 6))

    for j in range(len(mass_indexes)):
        i = mass_indexes[j]
        plt.plot(time[j] / yr, light_curve[j] / L_sun,
                 label=f'Total mass = {(m_target[i] + m_impactor[i]) / M_earth:.2f}' + ' $M_{\oplus}$', c=viridis(j / len(mass_indexes)))
    plt.xlabel('Time (yr)')
    plt.ylabel('Luminosity ($L_{\odot}$)')
    plt.legend()

    plt.xlim([0, 40])
    plt.savefig('figures/mass_light_curve_very_long.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_light_curve_very_long.png', bbox_inches='tight')

    plt.xlim([0, 5])
    plt.savefig('figures/mass_light_curve_long.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_light_curve_long.png', bbox_inches='tight')

    plt.xlim([0, 1])
    plt.savefig('figures/mass_light_curve_short.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_light_curve_short.png', bbox_inches='tight')

    plt.xlim([0.05, 20])
    plt.xscale('log')
    # plt.yscale('log')
    plt.savefig('figures/mass_light_curve_log.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_light_curve_log.png', bbox_inches='tight')

    plt.close()

    total_mass = (m_target + m_impactor)
    plt.scatter(total_mass[mass_indexes] / M_earth, t_half / day, label='$t_{1/2}$')
    # plt.scatter(total_mass[mass_indexes] / M_earth, t_quarter / yr, label='$t_{1/4}$')
    # plt.scatter(total_mass[mass_indexes] / M_earth, t_tenth / yr, label='$t_{1/10}$')
    plt.xlabel('Total mass ($M_{\oplus}$)')
    plt.ylabel('Cooling time (days)')
    plt.legend()

    plt.savefig('figures/mass_timescales.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_timescales.png', bbox_inches='tight')
    plt.close()

    plt.scatter(Q, L0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Modified specific Impact Energy (J/kg)')
    plt.ylabel('Luminosity ($L_{\odot}$)')

    plt.savefig('figures/QL_mass_plot.png', bbox_inches='tight')
    plt.savefig('figures/QL_mass_plot.pdf', bbox_inches='tight')
    plt.close()

    plt.scatter(total_mass[mass_indexes] / M_earth, L0)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Total mass ($M_{\oplus}$)')
    plt.ylabel('Luminosity ($L_{\odot}$)')

    plt.savefig('figures/ML_mass_plot.png', bbox_inches='tight')
    plt.savefig('figures/ML_mass_plot.pdf', bbox_inches='tight')
    plt.close()


def changing_impact_parameter():
    Q = Q_prime[impact_parameter_indexes]
    L0, AM, SAM = [], [], []

    light_curve, time = [], []
    t_half, t_quarter, t_tenth = [], [], []

    for i in impact_parameter_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0.append(phot.luminosity / L_sun)
        t, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2(save_name=f'impact{i}', plot=False,
                                                                          plot_interval=0.1)
        light_curve.append(lum)
        time.append(t)
        AM.append(snap.total_angular_momentum)
        SAM.append(snap.total_specific_angular_momentum)
        t_half.append(t2)
        t_quarter.append(t4)
        t_tenth.append(t10)

    t_half, t_quarter, t_tenth = np.array(t_half), np.array(t_quarter), np.array(t_tenth)

    plt.figure(figsize=(8, 6))

    for j in range(len(impact_parameter_indexes)):
        i = impact_parameter_indexes[j]
        plt.plot(time[j] / yr, light_curve[j] / L_sun,
                 label=f'Impact parameter = {impact_parameter[i]:.2f}',
                 c=viridis(j / len(impact_parameter_indexes)))
    plt.xlabel('Time (yr)')
    plt.ylabel('Luminosity ($L_{\odot}$)')
    plt.legend()

    plt.xlim([0.05, 20])
    plt.xscale('log')
    #plt.yscale('log')
    plt.savefig('figures/b_light_curve_log.pdf', bbox_inches='tight')
    plt.savefig('figures/b_light_curve_log.png', bbox_inches='tight')

    plt.close()

    plt.scatter(impact_parameter[impact_parameter_indexes], t_half / day, label='$t_{1/2}$')
    # plt.scatter(impact_parameter[impact_parameter_indexes], t_quarter / yr, label='$t_{1/4}$')
    # plt.scatter(impact_parameter[impact_parameter_indexes], t_tenth / yr, label='$t_{1/10}$')
    plt.xlabel('Impact parameter')
    plt.ylabel('Cooling time (days)')
    plt.yscale('log')

    plt.savefig('figures/b_timescales.pdf', bbox_inches='tight')
    plt.savefig('figures/b_timescales.png', bbox_inches='tight')
    plt.close()

    plt.scatter(Q, L0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Modified specific Impact Energy (J/kg)')
    plt.ylabel('Luminosity ($L_{\odot}$)')

    plt.savefig('figures/QL_b_plot.png', bbox_inches='tight')
    plt.savefig('figures/QL_b_plot.pdf', bbox_inches='tight')
    plt.close()

    plt.scatter(impact_parameter[impact_parameter_indexes], L0)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Impact parameter')
    plt.ylabel('Luminosity ($L_{\odot}$)')

    plt.savefig('figures/bL_plot.png', bbox_inches='tight')
    plt.savefig('figures/bL_plot.pdf', bbox_inches='tight')
    plt.close()


def changing_mass_with_impact_parameter():
    Q = Q_prime[mass_impact_parameter_indexes]
    L0, AM, SAM = [], [], []

    light_curve, time = [], []
    t_half, t_quarter, t_tenth = [], [], []

    for i in mass_impact_parameter_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0.append(phot.luminosity / L_sun)
        t, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2(save_name=f'impact{i}', plot=False,
                                                                          plot_interval=0.1)
        light_curve.append(lum)
        time.append(t)
        AM.append(snap.total_angular_momentum)
        SAM.append(snap.total_specific_angular_momentum)
        t_half.append(t2)
        t_quarter.append(t4)
        t_tenth.append(t10)

    t_half, t_quarter, t_tenth = np.array(t_half), np.array(t_quarter), np.array(t_tenth)

    plt.figure(figsize=(8, 6))

    for j in range(len(mass_impact_parameter_indexes)):
        i = mass_impact_parameter_indexes[j]
        plt.plot(time[j] / yr, light_curve[j] / L_sun,
                 label=f'Total mass = {(m_target[i] + m_impactor[i]) / M_earth:.2f}' + '$M_{\oplus}$',
                 c=viridis(j / len(mass_impact_parameter_indexes)))
    plt.xlabel('Time (yr)')
    plt.ylabel('Luminosity ($L_{\odot}$)')
    plt.legend()

    plt.xlim([0.05, 20])
    plt.xscale('log')
    # plt.yscale('log')
    plt.savefig('figures/mass_b0.1_light_curve_log.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_b0.1_light_curve_log.png', bbox_inches='tight')

    plt.close()

    total_mass = m_target + m_impactor
    plt.scatter(total_mass[mass_impact_parameter_indexes] / M_earth, t_half / day, label='$t_{1/2}$')
    # plt.scatter(total_mass[mass_impact_parameter_indexes] / M_earth, t_quarter / yr, label='$t_{1/4}$')
    # plt.scatter(total_mass[mass_impact_parameter_indexes] / M_earth, t_tenth / yr, label='$t_{1/10}$')
    plt.xlabel('Total mass ($M_{\oplus}$)')
    plt.ylabel('Time (day)')
    plt.legend()

    plt.savefig('figures/mass_b0.1_timescales.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_b0.1_timescales.png', bbox_inches='tight')
    plt.close()

    plt.scatter(Q, L0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Modified specific Impact Energy (J/kg)')
    plt.ylabel('Luminosity ($L_{\odot}$)')

    plt.savefig('figures/QL_b0.1_mass_plot.png', bbox_inches='tight')
    plt.savefig('figures/QL_b0.1_mass_plot.pdf', bbox_inches='tight')
    plt.close()

    plt.scatter(total_mass[mass_impact_parameter_indexes] / M_earth, L0)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Total mass ($M_{\oplus}$)')
    plt.ylabel('Luminosity ($L_{\odot}$)')

    plt.savefig('figures/ML_b0.1_mass_plot.png', bbox_inches='tight')
    plt.savefig('figures/ML_b0.1_mass_plot.pdf', bbox_inches='tight')
    plt.close()


def changing_mass_with_mass_ratio():
    Q = Q_prime[mass_mass_ratio_indexes]
    L0, AM, SAM = [], [], []

    light_curve, time = [], []
    t_half, t_quarter, t_tenth = [], [], []

    for i in mass_mass_ratio_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0.append(phot.luminosity / L_sun)
        t, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2(save_name=f'impact{i}', plot=False,
                                                                          plot_interval=0.1)
        light_curve.append(lum)
        time.append(t)
        AM.append(snap.total_angular_momentum)
        SAM.append(snap.total_specific_angular_momentum)
        t_half.append(t2)
        t_quarter.append(t4)
        t_tenth.append(t10)

    t_half, t_quarter, t_tenth = np.array(t_half), np.array(t_quarter), np.array(t_tenth)

    plt.figure(figsize=(8, 6))

    for j in range(len(mass_mass_ratio_indexes)):
        i = mass_mass_ratio_indexes[j]
        plt.plot(time[j] / yr, light_curve[j] / L_sun,
                 label=f'Total mass = {(m_target[i] + m_impactor[i]) / M_earth:.2f}' + '$M_{\oplus}$',
                 c=viridis(j / len(mass_mass_ratio_indexes)))
    plt.xlabel('Time (yr)')
    plt.ylabel('Luminosity ($L_{\odot}$)')
    plt.legend()

    plt.xlim([0.05, 20])
    plt.xscale('log')
    # plt.yscale('log')
    plt.savefig('figures/mass_r0.5_light_curve_log.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_r0.5_light_curve_log.png', bbox_inches='tight')

    plt.close()

    total_mass = m_target + m_impactor
    plt.scatter(total_mass[mass_mass_ratio_indexes] / M_earth, t_half / day, label='$t_{1/2}$')
    # plt.scatter(total_mass[mass_impact_parameter_indexes] / M_earth, t_quarter / yr, label='$t_{1/4}$')
    # plt.scatter(total_mass[mass_impact_parameter_indexes] / M_earth, t_tenth / yr, label='$t_{1/10}$')
    plt.xlabel('Total mass ($M_{\oplus}$)')
    plt.ylabel('Time (day)')

    plt.savefig('figures/mass_r0.5_timescales.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_r0.5_timescales.png', bbox_inches='tight')
    plt.close()

    plt.scatter(total_mass[mass_mass_ratio_indexes] / M_earth, L0)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Total mass ($M_{\oplus}$)')
    plt.ylabel('Luminosity ($L_{\odot}$)')

    plt.savefig('figures/ML_r0.5_mass_plot.png', bbox_inches='tight')
    plt.savefig('figures/ML_r0.5_mass_plot.pdf', bbox_inches='tight')
    plt.close()


def mass_luminosity_plots():
    L0, L0_b, L0_r = [], [], []
    t_cool, t_cool_b, t_cool_r = [], [], []

    for i in mass_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0.append(phot.luminosity / L_sun)

        t, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2()
        t_cool.append(t2)

    for i in mass_impact_parameter_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0_b.append(phot.luminosity / L_sun)

        t, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2()
        t_cool_b.append(t2)

    for i in mass_mass_ratio_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0_r.append(phot.luminosity / L_sun)

        t, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2()
        t_cool_r.append(t2)

    t_cool = np.array(t_cool)
    t_cool_b = np.array(t_cool_b)
    t_cool_r = np.array(t_cool_r)

    total_mass = m_target + m_impactor
    plt.scatter(total_mass[mass_indexes] / M_earth, L0, label='b = 0.5, $\gamma$ = 1.0')
    plt.scatter(total_mass[mass_impact_parameter_indexes] / M_earth, L0_b, label='b = 0.1, $\gamma$ = 1.0')
    plt.scatter(total_mass[mass_mass_ratio_indexes] / M_earth, L0_r, label='b = 0.5, $\gamma$ = 0.5')
    plt.xlabel('Total mass ($M_{\oplus}$)')
    plt.ylabel('Luminosity ($L_{\odot}$)')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()

    plt.savefig('figures/ML_full_plot.png', bbox_inches='tight')
    plt.savefig('figures/ML_full_plot.pdf', bbox_inches='tight')
    plt.close()

    plt.scatter(total_mass[mass_indexes] / M_earth, t_cool / day, label='b = 0.5, $\gamma$ = 1.0')
    plt.scatter(total_mass[mass_impact_parameter_indexes] / M_earth, t_cool_b / day, label='b = 0.1, $\gamma$ = 1.0')
    plt.scatter(total_mass[mass_mass_ratio_indexes] / M_earth, t_cool_r / day, label='b = 0.5, $\gamma$ = 0.5')
    plt.xlabel('Total mass ($M_{\oplus}$)')
    plt.ylabel('Cooling time (days)')
    plt.ylim(10, 1000)
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend()

    plt.savefig('figures/Mt_full_plot.png', bbox_inches='tight')
    plt.savefig('figures/Mt_full_plot.pdf', bbox_inches='tight')
    plt.close()


def generate_table():

    indexes = [0] + mass_indexes + impact_parameter_indexes + mass_impact_parameter_indexes + mass_mass_ratio_indexes
    L0 = np.zeros_like(Q_prime)
    cool_time = np.zeros_like(Q_prime)

    # removes duplicates
    result = []
    for i in indexes:
        if i not in result:
            result.append(i)

    indexes = result

    for i in indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0[i] = phot.luminosity / L_sun

        t, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2()

        cool_time[i] = t2 / day

    mass_ratio = m_impactor / m_target
    for j in range(len(indexes)):
        i = indexes[j]
        print(f'{j} & '
              f'{m_target[i] / M_earth:.1f} & '
              f'{impact_parameter[i]:.1f} & '
              f'{mass_ratio[i]:.2f} & '
              f'{v_over_v_esc[i]:.1f} & '
              f'{L0[i]/0.001:.1f} & '
              f'{cool_time[i]:.1g}\\\\')


def single_analysis(i):
    filename = get_filename(i, 4)
    snap = snapshot(filename)
    # s = gas_slice(snap, size=20)
    # s.full_plot()
    phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, 500, n_theta=40)

    # phot.plot('rho', plot_photosphere=True)
    # phot.plot('T', log=False, round_to=1000, plot_photosphere=True, val_max=8000)
    # phot.plot('P', plot_photosphere=True)
    # phot.plot('s', log=False, round_to=1000, plot_photosphere=True)

    phot.set_up()

    try:
        phot.plot('rho', plot_photosphere=True)
        phot.plot('T', log=False, round_to=1000, plot_photosphere=True, val_max=8000)
        phot.plot('P', plot_photosphere=True)
        phot.plot('phase', log=False, round_to=1, plot_photosphere=True)
        phot.plot('tau', plot_photosphere=True)
    except ValueError:
        plt.close()

    time, lum, A, R, T, m_dot, t_half, t_quarter, t_tenth = phot.long_term_evolution_v2()
    plt.plot(time / yr, lum / L_sun)
    plt.show()

    plt.plot(time / yr, R)
    plt.show()

    plt.plot(time / yr, A)
    plt.show()

    plt.plot(time / yr, T)
    plt.show()


def impact_plots():
    sims = [0, 2, 5, 10, 15, 19]
    labels = ['$M_{target}$ = 0.5 $M_{\oplus}$\n$\gamma$ = 1\nb = 0.5',
              '$M_{target}$ $\\rightarrow$ 0.25 $M_{\oplus}$',
              '$M_{target}$ $\\rightarrow$ 1.0 $M_{\oplus}$',
              '$\gamma$ $\\rightarrow$ 0.5',
              'b $\\rightarrow$ 0.1',
              'b $\\rightarrow$ 0.6']
    rows, cols = len(sims), len(snapshot_names)

    fig, ax = plt.subplots(nrows=rows, ncols=cols, squeeze=False, sharey='row', sharex='col')

    fig.set_figwidth(2.5 * cols)
    fig.set_figheight(2.5 * rows)

    for i in range(rows):
        for j in range(cols):
            snap = snapshot(get_filename(sims[i], j))
            img = gas_slice(snap, size=6)
            ax[i, j] = img.plot('rho', show=False, threshold=1e-2, ax=ax[i, j], colorbar=False)
            if i != rows - 1:
                ax[i, j].set_xlabel('')
                ax[i, j].set_xticklabels([])
            if j == 0:
                ax[i, j].annotate(labels[i], (0.1, 0.7) if i == 0 else (0.1, 0.9), xycoords='axes fraction',
                                  color='white')
            if i == 0:
                ax[i, j].set_title(f'T = {snapshot_times[j]:.1f} hrs')

    plt.subplots_adjust(wspace=0.02)
    plt.subplots_adjust(hspace=0.02)

    plt.savefig('figures/impact_plot_v2.png', bbox_inches='tight')
    plt.savefig('figures/impact_plot_v2.pdf', bbox_inches='tight')
    plt.close()


def example_extrapolation():

    filename = get_filename(3, 4)
    snap = snapshot(filename, plot_rotation=True)
    s = gas_slice(snap, size=18)
    R, theta = snap.HD_limit_R.value / 6371000, np.linspace(0, 2 * np.pi, num=1000)
    curve = [R * np.cos(theta), R * np.sin(theta)]
    s.plot('rho', threshold=1e-6, save='density_floor', show=False, curve=curve, curve_label='Extrapolation point')
    phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, 1600, n_theta=100)
    phot.set_up()
    phot.plot('P', plot_photosphere=True, save='P_extrapolation', val_min=100, val_max=1e10, ylim=[-25, 25],
              xlim=[0, 50], cmap='inferno')


def phase_diagram(filename):
    S = np.linspace(2500, 12000, num=100)
    P = np.logspace(1, 9, num=100)
    x, y = np.meshgrid(S, P)
    rho, T = fst.rho_EOS(x, y), fst.T1_EOS(x, y)
    z = np.log10(fst.alpha(rho, T, y, x))
    z = np.where(np.isfinite(z), z, np.NaN)
    plt.figure(figsize=(13, 9))
    CS = plt.contourf(x, y, z, 200, cmap='viridis', rasterized=True)

    for a in CS.collections:
        a.set_edgecolor("face")

    cbar = plt.colorbar(label='$\log_{10}$[' + data_labels['alpha'] + ']')

    tick_positions = np.arange(np.ceil(np.nanmin(z)), np.ceil(np.nanmax(z)), 2)
    cbar.set_ticks(tick_positions)

    plt.yscale('log')
    plt.xlabel(data_labels['s'])
    plt.ylabel(data_labels['P'])
    plt.plot(fst.NewEOS.vc.Sl, fst.NewEOS.vc.Pl, 'w-', label='Vapor Dome')
    plt.plot(fst.NewEOS.vc.Sv, fst.NewEOS.vc.Pv, 'w-')
    plt.vlines(fst.S_critical_point, fst.P_critical_point, 1e9, colors='white')
    plt.scatter(fst.S_critical_point, fst.P_critical_point, c='white', label='Critical Point')
    plt.xlim([4000, 12000])
    plt.ylim([1e1, 1e9])
    plt.annotate('Liquid', (4400, 1e8), c='black')
    plt.annotate('Liquid + Vapour', (6300, 1e4), c='black')
    plt.annotate('Vapour', (9800, 1e8), c='black')

    snap = snapshot(filename)
    phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, 600, n_theta=40)

    S, P = phot.data['s'][20, :], phot.data['P'][20, :]
    plt.plot(S, P, color='black', linestyle='--', label='Thermal profile with droplets')

    rad = np.array([2, 4, 6, 8, 10, 12, 15, 20, 30])
    r_labels = []
    S_points, P_points = np.zeros_like(rad), np.zeros_like(rad)
    for i in range(len(rad)):
        r_labels.append(f'{rad[i]}' + ' $R_{\oplus}$')
        j = phot.get_index(rad[i] * 6371000, 0)[1]
        S_points[i], P_points[i] = phot.data['s'][20, j], phot.data['P'][20, j]

    plt.scatter(S_points, P_points, color='black', s=8, marker='o')
    for j in range(len(rad)):
        plt.annotate(r_labels[j], (S_points[j], P_points[j]), xytext=(-37, -5), textcoords='offset points',
                     color='black')

    phot.remove_droplets()
    S, P = phot.data['s'][20, :], phot.data['P'][20, :]
    plt.plot(S, P, color='red', linestyle='-.', label='Thermal profile without droplets')

    plt.legend(loc='lower left')
    plt.savefig('figures/phase_diagram.png', bbox_inches='tight')
    plt.savefig('figures/phase_diagram.pdf', bbox_inches='tight')
    plt.close()


def test_hill_sphere():

    L0_15, L0_30, L0_60 = [], [], []
    L0_15_no_infall, L0_30_no_infall, L0_60_no_infall = [], [], []

    t_cool_15, t_cool_30, t_cool_60 = [], [], []
    t_cool_15_no_infall, t_cool_30_no_infall, t_cool_60_no_infall = [], [], []

    for i in mass_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot_15 = photosphere(snap, 12 * Rearth, 15 * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot_30 = photosphere(snap, 12 * Rearth, 30 * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot_60 = photosphere(snap, 12 * Rearth, 60 * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot_15_no_infall = photosphere(snap, 12 * Rearth, 15 * Rearth, resolution, n_theta=n_theta, n_phi=n_phi, droplet_infall=False)
        phot_30_no_infall = photosphere(snap, 12 * Rearth, 30 * Rearth, resolution, n_theta=n_theta, n_phi=n_phi, droplet_infall=False)
        phot_60_no_infall = photosphere(snap, 12 * Rearth, 60 * Rearth, resolution, n_theta=n_theta, n_phi=n_phi, droplet_infall=False)

        phot_15.set_up()
        phot_30.set_up()
        phot_60.set_up()
        phot_15_no_infall.set_up()
        phot_30_no_infall.set_up()
        phot_60_no_infall.set_up()

        L0_15.append(phot_15.luminosity / L_sun)
        t, lum, A, R, T, m_dot, t2, t4, t10 = phot_15.long_term_evolution_v2()
        t_cool_15.append(t2 / day)

        L0_30.append(phot_30.luminosity / L_sun)
        t, lum, A, R, T, m_dot, t2, t4, t10 = phot_30.long_term_evolution_v2()
        t_cool_30.append(t2 / day)

        L0_60.append(phot_60.luminosity / L_sun)
        t, lum, A, R, T, m_dot, t2, t4, t10 = phot_60.long_term_evolution_v2()
        t_cool_60.append(t2 / day)

        L0_15_no_infall.append(phot_15_no_infall.luminosity / L_sun)
        t, lum, A, R, T, m_dot, t2, t4, t10 = phot_15_no_infall.long_term_evolution_v2()
        t_cool_15_no_infall.append(t2 / day)

        L0_30_no_infall.append(phot_30_no_infall.luminosity / L_sun)
        t, lum, A, R, T, m_dot, t2, t4, t10 = phot_30_no_infall.long_term_evolution_v2()
        t_cool_30_no_infall.append(t2 / day)

        L0_60_no_infall.append(phot_60_no_infall.luminosity / L_sun)
        t, lum, A, R, T, m_dot, t2, t4, t10 = phot_60_no_infall.long_term_evolution_v2()
        t_cool_60_no_infall.append(t2 / day)

    total_mass = (m_target + m_impactor)
    mass = list(total_mass[mass_indexes] / M_earth)

    plt.scatter(mass, L0_15, c='indigo', marker='o', label='Hill sphere = 15 $R_{\oplus}$')
    plt.scatter(mass, L0_15_no_infall, c='indigo', marker='x', label='Hill sphere = 15 $R_{\oplus}$ (no droplet infall)')
    plt.scatter(mass, L0_30, c='darkcyan', marker='o', label='Hill sphere = 30 $R_{\oplus}$')
    plt.scatter(mass, L0_30_no_infall, c='darkcyan', marker='x', label='Hill sphere = 30 $R_{\oplus}$ (no droplet infall)')
    plt.scatter(mass, L0_60, c='gold', marker='o', label='Hill sphere = 60 $R_{\oplus}$')
    plt.scatter(mass, L0_60_no_infall, c='gold', marker='x', label='Hill sphere = 60 $R_{\oplus}$ (no droplet infall)')

    plt.xlabel('Total mass ($M_{\oplus}$)')
    plt.ylabel('Luminosity ($L_{\odot}$)')
    plt.legend()

    plt.savefig('figures/ML_hill_plot.png', bbox_inches='tight')
    plt.savefig('figures/ML_hill_plot.pdf', bbox_inches='tight')
    plt.close()

    plt.scatter(mass, t_cool_15, c='indigo', marker='o', label='Hill sphere = 15 $R_{\oplus}$')
    plt.scatter(mass, t_cool_15_no_infall, c='indigo', marker='x', label='Hill sphere = 15 $R_{\oplus}$ (no droplet infall)')
    plt.scatter(mass, t_cool_30, c='darkcyan', marker='o', label='Hill sphere = 30 $R_{\oplus}$')
    plt.scatter(mass, t_cool_30_no_infall, c='darkcyan', marker='x', label='Hill sphere = 30 $R_{\oplus}$ (no droplet infall)')
    plt.scatter(mass, t_cool_60, c='gold', marker='o', label='Hill sphere = 60 $R_{\oplus}$')
    plt.scatter(mass, t_cool_60_no_infall, c='gold', marker='x', label='Hill sphere = 60 $R_{\oplus}$ (no droplet infall)')

    plt.xlabel('Total mass ($M_{\oplus}$)')
    plt.ylabel('Cooling time (days)')
    #plt.legend()
    plt.yscale('log')

    plt.savefig('figures/Mt_hill_plot.png', bbox_inches='tight')
    plt.savefig('figures/Mt_hill_plot.pdf', bbox_inches='tight')
    plt.close()


def full_plot():
    L0_m, L0_b, L0_mb, L0_mr = [], [], [], []
    t_cool_m, t_cool_b, t_cool_mb, t_cool_mr = [], [], [], []

    for i in mass_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0_m.append(phot.luminosity / L_sun)

        t, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2()
        t_cool_m.append(t2)

    for i in impact_parameter_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0_b.append(phot.luminosity / L_sun)
        t, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2()
        t_cool_b.append(t2)

    for i in mass_impact_parameter_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0_mb.append(phot.luminosity / L_sun)

        t, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2()
        t_cool_mb.append(t2)

    for i in mass_mass_ratio_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, resolution, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0_mr.append(phot.luminosity / L_sun)

        t, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2()
        t_cool_mr.append(t2)

    t_cool_m = np.array(t_cool_m)
    t_cool_b = np.array(t_cool_b)
    t_cool_mb = np.array(t_cool_mb)
    t_cool_mr = np.array(t_cool_mr)

    L0_m = np.array(L0_m)
    L0_b = np.array(L0_b)
    L0_mb = np.array(L0_mb)
    L0_mr = np.array(L0_mr)

    fig, axs = plt.subplots(2, 2)
    fig.set_figwidth(16)
    fig.set_figheight(12)
    plt.subplots_adjust(hspace=0, wspace=0)

    total_mass = m_target + m_impactor

    axs[0, 0].scatter(total_mass[mass_indexes] / M_earth, L0_m, c='#E69F00', label='b = 0.5, $\gamma$ = 1.0')
    axs[0, 0].scatter(total_mass[mass_impact_parameter_indexes] / M_earth, L0_mb, c='#56B4E9', label='b = 0.1, $\gamma$ = 1.0')
    axs[0, 0].scatter(total_mass[mass_mass_ratio_indexes] / M_earth, L0_mr, c='#009E73', label='b = 0.5, $\gamma$ = 0.5')

    axs[1, 0].scatter(total_mass[mass_indexes] / M_earth, t_cool_m / day, c='#E69F00', label='b = 0.5, $\gamma$ = 1.0')
    axs[1, 0].scatter(total_mass[mass_impact_parameter_indexes] / M_earth, t_cool_mb / day, c='#56B4E9', label='b = 0.1, $\gamma$ = 1.0')
    axs[1, 0].scatter(total_mass[mass_mass_ratio_indexes] / M_earth, t_cool_mr / day, c='#009E73', label='b = 0.5, $\gamma$ = 0.5')

    axs[0, 1].scatter(impact_parameter[impact_parameter_indexes], L0_b, c='#CC79A7', label='$M_{\mathrm{total}}$ = 1.0 $M_{\oplus}$, $\gamma$ = 1.0')

    axs[1, 1].scatter(impact_parameter[impact_parameter_indexes], t_cool_b / day, c='#CC79A7')

    axs[0, 0].legend()
    axs[0, 1].legend()

    axs[0, 0].set_ylim([0, 0.04])
    axs[0, 1].set_ylim([0, 0.04])
    axs[1, 0].set_ylim([0, 650])
    axs[1, 1].set_ylim([0, 650])

    axs[0, 0].set_xlim([0, 4.4])
    axs[1, 0].set_xlim([0, 4.4])
    axs[0, 1].set_xlim([0, 0.65])
    axs[1, 1].set_xlim([0, 0.65])

    axs[0, 0].set_ylabel('Initial luminosity ($L_{\odot}$)')
    axs[0, 1].set_yticklabels([])
    axs[1, 0].set_ylabel('Cooling time (day)')
    axs[1, 1].set_yticklabels([])

    axs[1, 0].set_xlabel('Total mass ($M_{\oplus}$)')
    axs[0, 0].set_xticklabels([])
    axs[1, 1].set_xlabel('Impact parameter')
    axs[0, 1].set_xticklabels([])

    plt.savefig('figures/big_plot.png', bbox_inches='tight')
    plt.savefig('figures/big_plot.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    generate_table()
