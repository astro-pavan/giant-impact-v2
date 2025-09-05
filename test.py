# type: ignore

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import re
from tqdm import tqdm

from photosphere import photosphere, M_earth, L_sun, yr, day, R_earth
import EOS as fst

snapshot_path = '/data/pt426/Impact_sims/Final_Sims/'

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
final_mass = np.zeros(n_sims)
final_AM = np.zeros(n_sims)

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

total_mass = m_target + m_impactor
mass_ratio = m_impactor / total_mass


# gets the filename of a simulation from the simulation index and time index
def get_filename(i, i_time):
    return f'{snapshot_path}{directory[i]}{sim_name[i]}/{snapshot_names[i_time]}'

def results_plot_and_table():

    index = []
    total_m = []

    L0_1 = []
    t_cool_1 = []

    for i in tqdm([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23]):

        index.append(i)
        p1 = photosphere(get_filename(i, 4), orbital_period=3 * day)

        final_mass[i] = p1.snapshot.total_mass / M_earth
        final_AM[i] = p1.snapshot.total_angular_momentum
        total_m.append(total_mass[i])

        L0_1.append(p1.L_phot)
        t, L, R, T, t_half, t_tenth = p1.cool(20 * yr, n=10000)
        t_cool_1.append(t_half)

    L0_1 = np.array(L0_1) / L_sun
    t_cool_1 = np.array(t_cool_1)

    L0_10 = []
    t_cool_10 = []

    for i in tqdm([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23]):

        p1 = photosphere(get_filename(i, 4), orbital_period=30 * day)

        L0_10.append(p1.L_phot)
        t, L, R, T, t_half, t_tenth = p1.cool(80 * yr, n=10000)
        # plt.plot(t / yr, L / L_sun)
        # plt.ylim([0, L[0] / L_sun])
        # plt.title(i)
        # plt.savefig(f'{i}.png')
        # plt.close()

        # final_mass[i] = p1.snapshot.total_mass / M_earth
        # final_AM[i] = p1.snapshot.total_angular_momentum
        # total_m.append(total_mass[i])

        t_cool_10.append(t_half)

    L0_10 = np.array(L0_10) / L_sun
    t_cool_10 = np.array(t_cool_10)

    L0_100 = []
    t_cool_100 = []

    for i in tqdm([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23]):

        p1 = photosphere(get_filename(i, 4), orbital_period=300 * day)

        L0_100.append(p1.L_phot)
        t, L, R, T, t_half, t_tenth = p1.cool(20 * yr, n=10000)
        t_cool_100.append(t_half)

    L0_100 = np.array(L0_100) / L_sun
    t_cool_100 = np.array(t_cool_100)

    plt.figure(figsize=(8, 6), dpi=300)

    scatter1 = plt.scatter(t_cool_1 / day, L0_1, s=20,
                           c=total_m, cmap='viridis', marker='o', label='Period = 3 days')
    plt.scatter(t_cool_10 / day, L0_10, s=20,
                c=total_m, cmap='viridis', marker='x', label='Period = 30 days')
    plt.scatter(t_cool_100 / day, L0_100, s=20,
                c=total_m, cmap='viridis', marker='+', label='Period = 300 days')

    plt.colorbar(scatter1, label='Total impact mass ($M_{\oplus}$)')
    plt.clim(0, 4.0)

    # plt.xlim([0, (np.max(t_cool) / day) + 50])
    plt.xlim([2, 10000])
    plt.ylim([1e-5, 2e-2])

    plt.xlabel('Cooling time (day)')
    plt.ylabel('Initial luminosity ($L_{\odot}$)')

    plt.yscale('log')
    plt.xscale('log')

    plt.axvline(30, linestyle='--', color='black', label='Gaia average cadence')
    plt.axvline(3, linestyle='--', color='red', label='LSST average cadence')

    plt.legend()

    plt.savefig('figures/big_plot_v2.png', bbox_inches='tight')
    plt.savefig('figures/big_plot_v2.pdf', bbox_inches='tight')
    plt.close()

    print('INITIAL CONDITIONS TABLE:')

    print('\multicolumn{2}{c|}{Simulation} & \multicolumn{5}{c}{Initial Parameters}\\\\')
    print('Set & \# & No. of particles & $M_{\mathrm{total}}$ / $M_{\oplus}$ & $\gamma$ & $b$ & $v_{\mathrm{imp}}$ / $v_{\mathrm{esc}}$ \\\\ \hline')

    for j in range(len(index)):
        i = index[j]
        print(' & ', end='')
        print(f'{j} & ', end='')
        print(f'{n_particles[i]:.0f} & ', end='')
        print(f'{total_mass[i]:.2f} & ', end='')
        print(f'{mass_ratio[i]:.2f} & ', end='')
        print(f'{impact_parameter[i]:.2f} & ', end='')
        print(f'{v_over_v_esc[i]:.1f} \\\\')

    print('RESULTS TABLE:')

    print('\multicolumn{2}{c|}{Simulation} & \multicolumn{2}{c|}{} & \multicolumn{2}{c|}{P = 3 day} & \multicolumn{2}{c|}{P = 30 days} & \multicolumn{2}{c}{P = 300 days} \\\\')
    print('Set & \# & $M_{\mathrm{final}}$ / $M_{\oplus}$ & $AM_{\mathrm{final}}$ / $10^{34}$ $\\text{kg}\, \\text{m}^2\\text{s}^{-1}$ & $L_{0}$ / $10^{-3}L_{\odot}$ & $t_{\mathrm{cool}}$ (days) & $L_{0}$ / $10^{-3}L_{\odot}$ & $t_{\mathrm{cool}}$ (days) & $L_{0}$ / $10^{-3}L_{\odot}$ & $t_{\mathrm{cool}}$ (days) \\\\ \hline')

    for j in range(len(index)):
        i = index[j]
        print(' & ', end='')
        print(f'{j} & ', end='')
        print(f'{final_mass[i] / M_earth:.2f} & ', end='')
        print(f'{final_AM[i] / 1e34:.2f} & ', end='')
        print(f'{L0_1[j] / 0.001:.1f} & ', end='')
        print(f'{t_cool_1[j] / day:.0f} & ', end='')
        print(f'{L0_10[j] / 0.001:.1f} & ', end='')
        print(f'{t_cool_10[j] / day:.0f} & ', end='')
        print(f'{L0_100[j] / 0.001:.1f} & ', end='')
        print(f'{t_cool_100[j] / day:.0f} \\\\')

def phase_diagram():
    
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

    cbar = plt.colorbar(label='$\log_{10}$[Absorption ($m^{-1}$)]')

    tick_positions = np.arange(np.ceil(np.nanmin(z)), np.ceil(np.nanmax(z)), 2)
    cbar.set_ticks(tick_positions)

    plt.yscale('log')
    plt.xlabel('Specific Entropy (J/K/kg)')
    plt.ylabel('Pressure (Pa)')
    plt.plot(fst.NewEOS.vc.Sl, fst.NewEOS.vc.Pl, 'w-', label='Vapor Dome')
    plt.plot(fst.NewEOS.vc.Sv, fst.NewEOS.vc.Pv, 'w-')
    plt.vlines(fst.S_critical_point, fst.P_critical_point, 1e9, colors='white')
    plt.scatter(fst.S_critical_point, fst.P_critical_point, c='white', label='Critical Point')
    plt.xlim([4000, 12000])
    plt.ylim([1e1, 1e9])
    plt.annotate('Liquid', (4400, 1e8), c='black')
    plt.annotate('Liquid + Vapour', (6300, 1e4), c='black')
    plt.annotate('Vapour', (9800, 1e8), c='black')

    phot = photosphere(get_filename(0, 4), orbital_period=100*day, remove_droplets=False)

    S, P = phot.s, phot.P
    plt.plot(S, P, color='black', linestyle='--', label='Initial thermal profile', zorder=5, dashes=[5, 5])

    rad = np.array([2, 5, 10, 20, 30])
    r_labels = []
    S_points, P_points = np.zeros_like(rad), np.zeros_like(rad)
    for i in range(len(rad)):
        r_labels.append(f'{rad[i]}' + ' $R_{\oplus}$')
        j = np.argmin(phot.r < rad[i] * R_earth)
        # j = phot.get_index(rad[i] * 6371000, 0)[1]
        S_points[i], P_points[i] = phot.s[j], phot.P[j]

    plt.scatter(S_points, P_points, color='black', s=8, marker='o', zorder=4)
    for j in range(len(rad)):
        xytext = (-37, -5) if j < 2 else (7, -5)
        plt.annotate(r_labels[j], (S_points[j], P_points[j]), xytext=xytext, textcoords='offset points',
                     color='black', zorder=10)

    # tau = phot.data['tau'][20, :]
    # j = np.argmax(tau < 2/3)
    # print(phot.data['R'][20, j] / 6371000)
    # S_phot, P_phot = phot.data['s'][20, j], phot.data['P'][20, j]
    # plt.scatter(S_phot, P_phot, marker='x', color='red', label='$\\tau$ = $\\frac{2}{3}$', zorder=6)

    # plt.arrow(7000, 4e6, 650, 0, color='red', width=10, head_width=40, head_length=30, length_includes_head=True)

    phot.remove_droplets(override=True)

    S, P = phot.s, phot.P
    plt.plot(S, P, color='darkorange', linestyle='--', label='Thermal profile after droplet removal', dashes=[5, 5])

    for i in range(50):
        phot.cool_step(0.01 * yr)
        phot.remove_droplets(override=True)

    S, P = phot.s, phot.P
    S, P = S[S > 6000], P[S > 6000]
    plt.plot(S, P, color='red', linestyle='--', label='Thermal profile after cooling', dashes=[5, 5], zorder=9)

    rad = np.array([5, 10, 20, 30])
    r_labels = []
    S_points, P_points = np.zeros_like(rad), np.zeros_like(rad)
    for i in range(len(rad)):
        r_labels.append(f'{rad[i]}' + ' $R_{\oplus}$')
        j = np.argmin(phot.r < rad[i] * R_earth)
        # j = phot.get_index(rad[i] * 6371000, 0)[1]
        S_points[i], P_points[i] = phot.s[j], phot.P[j]

    plt.scatter(S_points, P_points, color='red', s=8, marker='o', zorder=4)
    for j in range(len(rad)):
        plt.annotate(r_labels[j], (S_points[j], P_points[j]), xytext=(-37, -5), textcoords='offset points',
                     color='black', zorder=10)

    plt.legend(loc='lower left')
    plt.savefig('figures/phase_diagram.png', bbox_inches='tight')
    plt.savefig('figures/phase_diagram.pdf', bbox_inches='tight')
    plt.close()
    # phot.plot('P', plot_photosphere=True, val_min=1, val_max=1e12)

def results_no_droplet_removal():
    
    index = []
    L0 = []
    t_cool = []
    L0_no_remove = []
    t_cool_no_remove = []
    total_m = []

    for i in tqdm([0, 1, 2, 3, 4, 5, 6, 7]):

        index.append(i)
        p1 = photosphere(get_filename(i, 4), orbital_period=30 * day)
        p1_no_remove = photosphere(get_filename(i, 4), remove_droplets=False, orbital_period=30 * day)

        total_m.append(total_mass[i])

        L0.append(p1.L_phot)
        L0_no_remove.append(p1_no_remove.L_phot)

        t, L, R, T, t_half, t_tenth = p1.cool(80 * yr, n=10000)
        t_cool.append(t_half)
        t, L, R, T, t_half, t_tenth = p1_no_remove.cool(80 * yr, n=10000)
        t_cool_no_remove.append(t_half)

    L0 = np.array(L0) / L_sun
    t_cool = np.array(t_cool)

    L0_no_remove = np.array(L0_no_remove) / L_sun
    t_cool_no_remove = np.array(t_cool_no_remove)

    plt.figure(figsize=(8, 6), dpi=300)

    scatter1 = plt.scatter(t_cool / day, L0, s=20,
                           c=total_m, cmap='viridis', marker='o', label='With droplet removal')
    scatter2 = plt.scatter(t_cool_no_remove / day, L0_no_remove, s=20,
                           c=total_m, cmap='viridis', marker='x', label='No droplet removal')

    plt.colorbar(scatter1, label='Total impact mass ($M_{\oplus}$)')
    plt.clim(0, 4.0)

    plt.legend()

    plt.xlim([0, (np.max(t_cool_no_remove) / day) + 50])
    # plt.xlim([0.1, 10000])

    plt.xlabel('Cooling time (day)')
    plt.ylabel('Initial luminosity ($L_{\odot}$)')

    plt.yscale('log')

    plt.savefig('figures/droplet.png', bbox_inches='tight')
    plt.savefig('figures/droplet.pdf', bbox_inches='tight')
    plt.close()

def orbital_period_plot():

    periods = [10, 20, 50, 100, 200, 500, 1000]

    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.subplots_adjust(hspace=0)
    fig.set_figwidth(6.4)
    fig.set_figheight(10)

    impact_indexes = [2, 0, 5]
    impact_labels = ['0.5', '1.0', '2.0']

    # impact_indexes = [0]
    # impact_labels = ['1.0']

    for i in tqdm(range(len(impact_indexes))):

        L0 = []
        t_cool = []

        for T in periods:

            filename = get_filename(impact_indexes[i], 4)

            phot = photosphere(filename, orbital_period=T*day)

            L0.append(phot.L_phot / L_sun)

            t, L, R, T, t_half, t_tenth = phot.cool(20 * yr, n=10000)
            t_cool.append(t_half / day)

        label1 = impact_labels[i]
        colour = viridis(i / len(impact_indexes))

        ax[0].scatter(periods, L0, marker='o', c=colour, label=label1)
        ax[1].scatter(periods, t_cool, marker='o', c=colour, label=label1)

    ax[0].set_yscale('log')
    ax[0].set_ylabel('Initial Luminosity ($L_{\odot}$)')

    ax[0].set_xscale('log')

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    ax[1].set_xlabel('Orbital period (days)')
    ax[1].set_ylabel('Cooling time (days)')
    ax[0].legend(title='Total mass ($M_{\oplus}$)')

    # P = np.logspace(0, 2)
    # L = 0.0002 * P
    # T = 2000 / P

    # ax[0].plot(P, L, 'k--')
    # ax[1].plot(P, T, 'k--')

    plt.savefig('figures/hill_plot.png', bbox_inches='tight')
    plt.savefig('figures/hill_plot.pdf', bbox_inches='tight')
    plt.close()

def orbital_period_plot_v2():

    index = []
    total_m = []

    L0_1 = []
    t_cool_1 = []

    indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    for i in tqdm(indexes):

        index.append(i)
        p1 = photosphere(get_filename(i, 4), orbital_period=3 * day)

        L0_1.append(p1.L_phot)
        t, L, R, T, t_half, t_tenth = p1.cool(20 * yr, n=10000)
        t_cool_1.append(t_half)

    L0_1 = np.array(L0_1) / L_sun
    t_cool_1 = np.array(t_cool_1)

    L0_10 = []
    t_cool_10 = []

    for i in tqdm(indexes):

        p1 = photosphere(get_filename(i, 4), orbital_period=30 * day)

        L0_10.append(p1.L_phot)
        t, L, R, T, t_half, t_tenth = p1.cool(80 * yr, n=10000)
        # plt.plot(t / yr, L / L_sun)
        # plt.ylim([0, L[0] / L_sun])
        # plt.title(i)
        # plt.savefig(f'{i}.png')
        # plt.close()

        # final_mass[i] = p1.snapshot.total_mass / M_earth
        # final_AM[i] = p1.snapshot.total_angular_momentum
        # total_m.append(total_mass[i])

        t_cool_10.append(t_half)

    L0_10 = np.array(L0_10) / L_sun
    t_cool_10 = np.array(t_cool_10)

    L0_100 = []
    t_cool_100 = []

    for i in tqdm(indexes):

        p1 = photosphere(get_filename(i, 4), orbital_period=300 * day)

        L0_100.append(p1.L_phot)
        t, L, R, T, t_half, t_tenth = p1.cool(20 * yr, n=10000)
        t_cool_100.append(t_half)

    L0_100 = np.array(L0_100) / L_sun
    t_cool_100 = np.array(t_cool_100)

    plt.figure(figsize=(8, 6), dpi=300)

    scatter1 = plt.scatter(t_cool_1 / day, L0_1, s=20,
                           c=indexes, cmap='tab20', marker='o', label='Period = 3 days')
    plt.scatter(t_cool_10 / day, L0_10, s=20,
                c=indexes, cmap='tab20', marker='x', label='Period = 30 days')
    plt.scatter(t_cool_100 / day, L0_100, s=20,
                c=indexes, cmap='tab20', marker='+', label='Period = 300 days')

    # plt.colorbar(scatter1, label='Simulation Number')
    # plt.clim(0, 4.0)

    # plt.xlim([0, (np.max(t_cool) / day) + 50])
    plt.xlim([2, 10000])
    plt.ylim([1e-5, 2e-2])

    plt.xlabel('Cooling time (day)')
    plt.ylabel('Initial luminosity ($L_{\odot}$)')

    plt.yscale('log')
    plt.xscale('log')

    # plt.axvline(30, linestyle='--', color='black', label='Gaia average cadence')
    # plt.axvline(3, linestyle='--', color='red', label='LSST average cadence')

    plt.legend()

    plt.savefig('figures/hill_plot_v2.png', bbox_inches='tight')
    plt.savefig('figures/hill_plot_v2.pdf', bbox_inches='tight')
    plt.close()

def pressure_floor_plot():
    
    impact_indexes = [0, 5, 7]
    impact_labels = ['1.0', '2.0', '4.0']

    pressures = [1e15, 1e13, 1e11, 1e9, 1e7]

    plt.figure(figsize=(6.4, 5))

    for i in tqdm(range(len(impact_indexes))):

        L0 = []
        t_cool = []

        for P in pressures:

            # print(f'{i}: {P}')

            filename = get_filename(impact_indexes[i], 4)

            phot = photosphere(filename, pressure_floor=P)

            L0.append(phot.L_phot / L_sun)

            t, L, R, T, t_half, t_tenth = phot.cool(160 * yr, n=10000)
            t_cool.append(t_half / day)

            # plt.plot(t / yr, L / L_sun)
            # plt.ylim([0, L[0] / L_sun])
            # plt.title(f'{P:.1e}-{impact_labels[i]}.png')
            # plt.savefig(f'{P:.1e}-{impact_labels[i]}.png')
            # plt.close()

        label1 = impact_labels[i]
        colour = viridis(i / len(impact_indexes))

        plt.scatter(pressures, t_cool, marker='o', c=colour, label=label1)

    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel('Inner pressure limit (Pa)')
    plt.ylabel('Cooling time (days)')
    plt.legend(title='Total mass ($M_{\oplus}$)')

    plt.savefig('figures/pressure_plot.png', bbox_inches='tight')
    plt.savefig('figures/pressure_plot.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    orbital_period_plot_v2()
