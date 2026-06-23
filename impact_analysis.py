# type: ignore

# main analysis code
import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from matplotlib.colors import Normalize
from unyt import Rearth
import re
import pandas as pd
from tqdm import tqdm

from snapshot_analysis import snapshot, gas_slice, data_labels
from photosphere import photosphere, M_earth, L_sun, yr
import EOS as fst

day = 3600 * 24

hill_radius = 60
orbital_period = 1 * day
n_theta = 40
n_phi = 20
res = 1000


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

snapshot_path = '/home/pavan/MSci/Final_Sims/'

directory = ['',
             'target mass/', 'target mass/', 'target mass/', 'target mass/', 'target mass/', 'target mass/', 'target mass/',
             'mass ratio/', 'mass ratio/', 'mass ratio/',
             'mass ratio/target mass/', 'mass ratio/target mass/', 'mass ratio/target mass/', # 'mass ratio/target mass/',
             'impact parameter/', 'impact parameter/', 'impact parameter/', 'impact parameter/', 'impact parameter/', 'impact parameter/', # 'impact parameter/',
             'impact parameter/target mass/', 'impact parameter/target mass/', 'impact parameter/target mass/', 'impact parameter/target mass/']

sim_name = ['impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.50_spin0.0',  # 0

            'impact_p1.0e+05_M0.1_ratio1.00_v1.10_b0.50_spin0.0',  # 1
            'impact_p1.0e+05_M0.2_ratio1.00_v1.10_b0.50_spin0.0',  # 2
            'impact_p1.0e+05_M0.4_ratio1.00_v1.10_b0.50_spin0.0',  # 3
            'impact_p1.0e+05_M0.8_ratio1.00_v1.10_b0.50_spin0.0',  # 4
            'impact_p1.0e+05_M1.0_ratio1.00_v1.10_b0.50_spin0.0',  # 5
            'impact_p1.0e+05_M1.5_ratio1.00_v1.10_b0.50_spin0.0',  # 6
            'impact_p1.0e+05_M2.0_ratio1.00_v1.10_b0.50_spin0.0',  # 7

            'impact_p1.0e+05_M0.5_ratio0.05_v1.10_b0.50_spin0.0',  # 8
            'impact_p1.0e+05_M0.5_ratio0.20_v1.10_b0.50_spin0.0',  # 9
            'impact_p1.0e+05_M0.5_ratio0.50_v1.10_b0.50_spin0.0',  # 10

            'impact_p1.0e+05_M0.1_ratio0.50_v1.10_b0.50_spin0.0',  # 11
            'impact_p1.0e+05_M0.2_ratio0.50_v1.10_b0.50_spin0.0',  # 12
            # 'impact_p1.0e+05_M1.0_ratio0.50_v1.10_b0.50_spin0.0',  # 13
            'impact_p1.0e+05_M2.0_ratio0.50_v1.10_b0.50_spin0.0',  # 14

            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.10_spin0.0',  # 15
            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.20_spin0.0',  # 16
            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.30_spin0.0',  # 17
            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.40_spin0.0',  # 18
            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.60_spin0.0',  # 19
            'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.70_spin0.0',  # 20
            # 'impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.80_spin0.0',  # this one is a hit and run collision and is excluded

            'impact_p1.0e+05_M0.1_ratio1.00_v1.10_b0.10_spin0.0',  # 21
            'impact_p1.0e+05_M0.2_ratio1.00_v1.10_b0.10_spin0.0',  # 22
            'impact_p1.0e+05_M1.0_ratio1.00_v1.10_b0.10_spin0.0',  # 23
            'impact_p1.0e+05_M2.0_ratio1.00_v1.10_b0.10_spin0.0'   # 24
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
Q_prime = modified_specific_impact_energy(m_target * M_earth, m_impactor* M_earth, v_impact, impact_parameter)

mass_indexes = [1, 2, 3, 0, 4, 5, 6, 7]

impact_parameter_indexes = [15, 16, 17, 18, 0, 19, 20]
mass_ratio_indexes = [8, 9, 10, 0]

mass_mass_ratio_indexes = [11, 12, 10, 13, 14]
mass_impact_parameter_indexes = [21, 22, 15, 23, 24]


# gets the filename of a simulation from the simulation index and time index
def get_filename(i, i_time):
    return f'{snapshot_path}{directory[i]}{sim_name[i]}/{snapshot_names[i_time]}'


# produces a series of light curves for the giant impacts
def light_curves(indexes):

    L0 = []
    AM = []
    SAM = []

    light_curve = []
    time = []
    t_half = []

    for i in indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        #phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, resolution=res, n_theta=n_theta, n_phi=n_phi)
        phot = photosphere(snap, 12 * Rearth, resolution=res, period=orbital_period, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0.append(phot.luminosity / L_sun)
        t, lum, A, R, T, m_dot, t2, t10 = phot.long_term_evolution()
        light_curve.append(lum)
        time.append(t)
        AM.append(snap.total_angular_momentum)
        SAM.append(snap.total_specific_angular_momentum)
        t_half.append(t2)

    t_half = np.array(t_half)

    plt.figure(figsize=(8, 6))

    j = 0
    for i in indexes:
        plt.plot(time[j] / yr, light_curve[j] / L_sun,
                 label=f'{m_target[i] + m_impactor[i]:.2f}',
                 c=viridis(j / len(indexes)), linewidth=1.5)
        j += 1

    plt.xlabel('Time (days)')
    plt.ylabel(r'Luminosity ($L_{\odot}$)')
    plt.legend(title=r'Total mass ($M_{\oplus}$)')

    plt.ylim([0, 0.013])

    plt.xlim([0, 40])
    plt.savefig('figures/mass_light_curve_very_long.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_light_curve_very_long.png', bbox_inches='tight')

    plt.xlim([0, 2])
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


# generates the table that summarizes the initial conditions and results for each simulation
def generate_table():

    indexes = [0] # + mass_indexes + impact_parameter_indexes + mass_impact_parameter_indexes + mass_mass_ratio_indexes
    L0 = np.zeros_like(Q_prime)
    cool_time = np.zeros_like(Q_prime)
    final_mass = np.zeros_like(Q_prime)
    final_AM = np.zeros_like(Q_prime)

    # removes duplicate simulation indexes
    result = []
    for i in indexes:
        if i not in result:
            result.append(i)

    indexes = result

    for i in indexes:
        print(f'Simulation {i}:')
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        final_mass[i] = snap.total_mass
        final_AM[i] = snap.total_angular_momentum

        phot = photosphere(snap, 12 * Rearth, resolution=res, period=orbital_period, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0[i] = phot.luminosity / L_sun

        t, lum, A, R, T, m_dot, t2, t10 = phot.long_term_evolution()
        cool_time[i] = t2 / day

    print('DONE')

    m_total = m_target + m_impactor
    mass_ratio = m_impactor / m_total
    final_mass = final_mass / M_earth

    for j in range(len(indexes)):
        i = indexes[j]
        print(f'{j} &', end='')
        print(f'{n_particles[i]} & ', end='')
        print(f'{m_total[i]:.2f} & ', end='')
        print(f'{mass_ratio[i]:.2f} & ', end='')
        print(f'{v_over_v_esc[i]:.1f} & ', end='')
        print(f'{final_mass[i]:.2f} & ', end='')
        print(f'{final_AM[i] / 1e34:.2f} & ', end='')
        print(f'{L0[i] / 0.001:.1f} & ', end='')
        print(f'{cool_time[i]:.1e} \\\\')


# analyses a single snapshot
def single_analysis(filename):
    snap = snapshot(filename)
    # s = gas_slice(snap, size=20)
    # s.full_plot()
    phot = photosphere(snap, 12 * Rearth, hill_radius * Rearth, 2000, n_theta=40)

    # phot.plot('rho', plot_photosphere=True)
    # phot.plot('T', log=False, round_to=1000, plot_photosphere=True, val_max=8000)
    # phot.plot('P', plot_photosphere=True)
    # phot.plot('s', log=False, round_to=1000, plot_photosphere=True)

    phot.set_up()

    # phot.plot('rho', plot_photosphere=True)
    # phot.plot('T', log=False, round_to=1000, plot_photosphere=True, val_max=8000)
    # phot.plot('P', plot_photosphere=True)
    # phot.plot('t_infall', plot_photosphere=True, val_min=1e-2, val_max=1e8)
    plt.close()

    time, lum, A, R, T, m_dot, t_half, t_tenth = phot.long_term_evolution()
    plt.plot(time / yr, lum / L_sun)
    plt.show()

    # plt.plot(time / yr, R)
    # plt.show()
    #
    # plt.plot(time / yr, A)
    # plt.show()
    #
    # plt.plot(time / yr, T)
    # plt.show()


# produces a plot showing a series of impacts
def impact_plots():
    sims = [0, 2, 5, 10, 15, 19]
    labels = ['$M_{target}$ = 0.5 $M_{\\oplus}$\n$\\gamma$ = 0.50\nb = 0.5',
              r'$M_{target}$ $\rightarrow$ 0.25 $M_{\oplus}$',
              r'$M_{target}$ $\rightarrow$ 1.0 $M_{\oplus}$',
              r'$\gamma$ $\rightarrow$ 0.33',
              r'b $\rightarrow$ 0.1',
              r'b $\rightarrow$ 0.6']
    rows, cols = len(sims), len(snapshot_names)

    fig, ax = plt.subplots(nrows=rows, ncols=cols, squeeze=False, sharey='row', sharex='col')

    fig.set_figwidth(2.5 * cols)
    fig.set_figheight(2.5 * rows)

    for i in range(rows):
        for j in range(cols):
            snap = snapshot(get_filename(sims[i], j))
            img = gas_slice(snap, size=6)
            ax[i, j], im = img.plot('rho', show=False, threshold=1e1, ax=ax[i, j], colorbar=False, val_max=1e4)
            if i != rows - 1:
                ax[i, j].set_xlabel('')
                ax[i, j].set_xticklabels([])
            if j == 0:
                ax[i, j].annotate(labels[i], (0.1, 0.7) if i == 0 else (0.1, 0.9), xycoords='axes fraction',
                                  color='white')
            if i == 0:
                ax[i, j].set_title(f'T = {snapshot_times[j]:.1f} hrs')
                if j == 0:
                    cbaxes = fig.add_axes([0.93, 0.11, 0.03, 0.77])
                    fig.colorbar(im, label="Density ($kg/cm^{3}$)", ax=ax.ravel().tolist(), orientation='vertical', cax=cbaxes)

    plt.subplots_adjust(wspace=0.02)
    plt.subplots_adjust(hspace=0.02)

    plt.savefig('figures/impact_plot_v2.png', bbox_inches='tight')
    plt.savefig('figures/impact_plot_v2.pdf', bbox_inches='tight')
    plt.close()


# produces a series of plots that demonstrates some aspects of the extrapolation
def example_extrapolation(i):

    filename = get_filename(i, 4)
    snap = snapshot(filename, plot_rotation=True)
    s = gas_slice(snap, size=18)
    R, theta = snap.HD_limit_R.value / 6371000, np.linspace(0, 2 * np.pi, num=1000)
    curve = [R * np.cos(theta), R * np.sin(theta)]
    s.plot('rho', threshold=1e-6, save='density_floor', show=False, curve=curve, curve_label='Extrapolation point')
    phot = photosphere(snap, 12 * Rearth, resolution=1600, period=orbital_period, n_theta=100)
    phot.set_up(extra_cool=1e6)
    phot.plot('P', plot_photosphere=True, save='P_extrapolation', val_min=100, val_max=1e10, ylim=[-19, 19],
              xlim=[0, 38], cmap='inferno')
    phot.plot('tau', plot_photosphere=True, save='tau_extrapolation', val_min=1e-6, val_max=1e12, ylim=[-19, 19],
              xlim=[0, 38], cmap='inferno')
    phot.plot('P', plot_photosphere=True, save='P_extrapolation_2', val_min=100, val_max=1e10, ylim=[-30, 30],
              xlim=[0, 100], cmap='inferno')
    phot.plot('tau', plot_photosphere=True, save='tau_extrapolation_2', val_min=1e-6, val_max=1e12, ylim=[-30, 30],
              xlim=[0, 100], cmap='inferno')
    phot.plot('T', plot_photosphere=True, save='T_extrapolation', log=False, val_min=2000, val_max=12000, ylim=[-25, 25],
              xlim=[0, 100], cmap='plasma', round_to=1000)
    phot.plot('rho', plot_photosphere=True, save='rho_extrapolation', log=True, val_min=1e-10, val_max=1e2,
              ylim=[-25, 25], xlim=[0, 100], cmap='magma')


# produces a phase diagram with the thermal profile of a post impact body
def phase_diagram(filename, sim_index=None):
    S = np.linspace(2500, 12000, num=400)
    P = np.logspace(1, 9, num=400)
    x, y = np.meshgrid(S, P)
    rho, T = fst.rho_EOS(x, y), fst.T1_EOS(x, y)
    z = np.log10(fst.alpha(rho, T, y, x))
    z = np.where(np.isfinite(z), z, np.nan)
    plt.figure(figsize=(13, 9))
    CS = plt.pcolormesh(x, y, z, cmap='viridis', rasterized=True,
                        vmin=np.nanmin(z), vmax=np.nanmax(z))

    cbar = plt.colorbar(CS, label=r'$\log_{10}$[' + data_labels['alpha'] + ']')

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

    R_earth_m = 6.371e6  # m

    def radial_index(phot, r_re):
        return np.argmin(np.abs(phot.r - r_re * R_earth_m))

    phot = photosphere(filename, orbital_period=100 * day, remove_droplets=False)

    S, P = phot.s, phot.P
    plt.plot(S, P, color='black', linestyle='--', label='Initial thermal profile', zorder=5, alpha=0.9)

    rad = np.array([2, 5, 10, 20, 30], dtype=float)
    r_labels = [f'{int(r)}' + r' $R_{\oplus}$' for r in rad]
    S_points = np.array([phot.s[radial_index(phot, r)] for r in rad])
    P_points = np.array([phot.P[radial_index(phot, r)] for r in rad])

    plt.scatter(S_points, P_points, color='black', s=8, marker='o', zorder=4)
    for j in range(len(rad)):
        xytext = (-37, -5) if j < 2 else (7, -5)
        plt.annotate(r_labels[j], (S_points[j], P_points[j]), xytext=xytext, textcoords='offset points',
                     color='black', zorder=10)

    phot.remove_droplets(override=True)
    phot.calculate_luminosity()

    S, P = phot.s, phot.P
    plt.plot(S, P, color='darkorange', linestyle='-', label='Thermal profile after droplet removal', alpha=0.9)

    for i in range(50):
        phot.cool_step(0.01 * yr)
        phot.remove_droplets(override=True)
        phot.calculate_luminosity()

    S, P = phot.s, phot.P
    mask = S > 6000
    plt.plot(S[mask], P[mask], color='red', linestyle='--', label='Thermal profile after cooling', zorder=9, alpha=0.9)

    rad2 = np.array([5, 10, 20, 30], dtype=float)
    r_labels2 = [f'{int(r)}' + r' $R_{\oplus}$' for r in rad2]
    S_points2 = np.array([phot.s[radial_index(phot, r)] for r in rad2])
    P_points2 = np.array([phot.P[radial_index(phot, r)] for r in rad2])

    plt.scatter(S_points2, P_points2, color='red', s=8, marker='o', zorder=4)
    for j in range(len(rad2)):
        plt.annotate(r_labels2[j], (S_points2[j], P_points2[j]), xytext=(-37, -5), textcoords='offset points',
                     color='black', zorder=10)

    if sim_index is not None:
        plt.annotate(f'Simulation {sim_index}', (0.98, 0.98), xycoords='axes fraction',
                     ha='right', va='top', color='black', fontsize=11)

    plt.legend(loc='lower left')
    plt.savefig('figures/phase_diagram.png', bbox_inches='tight')
    plt.savefig('figures/phase_diagram.pdf', bbox_inches='tight')
    plt.close()


# produces a large plot that summarizes the simulation results
def full_plot():
    L0_m, L0_b, L0_mb, L0_mr = [], [], [], []
    t_cool_m, t_cool_b, t_cool_mb, t_cool_mr = [], [], [], []

    L0 = np.zeros_like(m_target)
    t_cool = np.zeros_like(m_target)
    final_mass = np.zeros_like(m_target)
    final_AM = np.zeros_like(m_target)

    indexes = mass_indexes + impact_parameter_indexes + mass_impact_parameter_indexes + mass_mass_ratio_indexes

    for i in mass_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        final_mass[i] = snap.total_mass
        final_AM[i] = snap.total_angular_momentum

        phot = photosphere(snap, 12 * Rearth, resolution=res, period=orbital_period, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0_m.append(phot.luminosity / L_sun)
        L0[i] = phot.luminosity / L_sun

        t, lum, A, R, T, m_dot, t2, t10 = phot.long_term_evolution()
        t_cool_m.append(t2)
        t_cool[i] = t2

    for i in impact_parameter_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        final_mass[i] = snap.total_mass
        final_AM[i] = snap.total_angular_momentum

        phot = photosphere(snap, 12 * Rearth, resolution=res, period=orbital_period, n_theta=n_theta, n_phi=n_phi)
        phot.set_up(extra_cool=144000)
        L0_b.append(phot.luminosity / L_sun)
        L0[i] = phot.luminosity / L_sun

        t, lum, A, R, T, m_dot, t2, t10 = phot.long_term_evolution()
        t_cool_b.append(t2)
        t_cool[i] = t2

    for i in mass_impact_parameter_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        final_mass[i] = snap.total_mass
        final_AM[i] = snap.total_angular_momentum

        phot = photosphere(snap, 12 * Rearth, resolution=res, period=orbital_period, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0_mb.append(phot.luminosity / L_sun)
        L0[i] = phot.luminosity / L_sun

        t, lum, A, R, T, m_dot, t2, t10 = phot.long_term_evolution()
        t_cool_mb.append(t2)
        t_cool[i] = t2

    for i in mass_mass_ratio_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        final_mass[i] = snap.total_mass
        final_AM[i] = snap.total_angular_momentum

        phot = photosphere(snap, 12 * Rearth, resolution=res, period=orbital_period, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0_mr.append(phot.luminosity / L_sun)
        L0[i] = phot.luminosity / L_sun

        t, lum, A, R, T, m_dot, t2, t10 = phot.long_term_evolution()
        t_cool_mr.append(t2)
        t_cool[i] = t2

    t_cool_m, t_cool_b, t_cool_mb, t_cool_mr = np.array(t_cool_m), np.array(t_cool_b), np.array(t_cool_mb), np.array(t_cool_mr)
    L0_m, L0_b, L0_mb, L0_mr = np.array(L0_m), np.array(L0_b), np.array(L0_mb), np.array(L0_mr)

    total_mass = m_target + m_impactor
    mass_ratio = m_impactor / total_mass

    plt.figure(figsize=(8, 6), dpi=300)

    scatter1 = plt.scatter(t_cool / day, L0, s=20,
                           c=total_mass, cmap='viridis', marker='o')

    plt.colorbar(scatter1, label=r'Total impact mass ($M_{\oplus}$)')
    plt.clim(0, 4.0)

    plt.xlim([0, (np.max(t_cool) / day) + 50])

    plt.xlabel('Cooling time (day)')
    plt.ylabel(r'Initial luminosity ($L_{\odot}$)')

    plt.yscale('log')

    plt.savefig('figures/big_plot_v2.png', bbox_inches='tight')
    plt.savefig('figures/big_plot_v2.pdf', bbox_inches='tight')
    plt.close()

    plt.style.use('default')

    for j in range(len(indexes)):
        i = indexes[j]
        print(f'{j} & ', end='')
        print(f'{n_particles[i]:.0f} & ', end='')
        print(f'{total_mass[i]:.2f} & ', end='')
        print(f'{mass_ratio[i]:.2f} & ', end='')
        print(f'{impact_parameter[i]:.2f} & ', end='')
        print(f'{v_over_v_esc[i]:.1f} & ', end='')
        print(f'{final_mass[i] / M_earth:.2f} & ', end='')
        print(f'{final_AM[i] / 1e34:.2f} & ', end='')
        print(f'{L0[i] / 0.001:.1f} & ', end='')
        print(f'{t_cool[i] / day:.0f} \\\\')


def compute_and_save_results():

    gaia_cadence = 30 * day
    lsst_cadence = 3 * day

    def do_all_with_period(period):

        index, L0, t_cool, total_mass, total_AM = [], [], [], [], []
        L_gaia, L_lsst = [], []

        for i in tqdm(mass_indexes):
            filename = get_filename(i, 4)

            phot = photosphere(filename, orbital_period=period)

            L0.append(phot.L_phot / L_sun)
            total_mass.append(phot.snapshot.total_mass / M_earth)
            total_AM.append(phot.snapshot.total_angular_momentum)
            index.append(i)

            t, lum, R, T, t2, t10 = phot.cool(100 * yr, n=1000)
            t_cool.append(t2 / day)
            L_gaia.append(float(np.interp(gaia_cadence, t, lum)) / L_sun)
            L_lsst.append(float(np.interp(lsst_cadence, t, lum)) / L_sun)

        for i in tqdm(mass_ratio_indexes):
            filename = get_filename(i, 4)

            phot = photosphere(filename, orbital_period=period)

            L0.append(phot.L_phot / L_sun)
            total_mass.append(phot.snapshot.total_mass / M_earth)
            total_AM.append(phot.snapshot.total_angular_momentum)
            index.append(i)

            t, lum, R, T, t2, t10 = phot.cool(100 * yr, n=1000)
            t_cool.append(t2 / day)
            L_gaia.append(float(np.interp(gaia_cadence, t, lum)) / L_sun)
            L_lsst.append(float(np.interp(lsst_cadence, t, lum)) / L_sun)

        for i in tqdm(impact_parameter_indexes):
            filename = get_filename(i, 4)

            phot = photosphere(filename, orbital_period=period)

            L0.append(phot.L_phot / L_sun)
            total_mass.append(phot.snapshot.total_mass / M_earth)
            total_AM.append(phot.snapshot.total_angular_momentum)
            index.append(i)

            t, lum, R, T, t2, t10 = phot.cool(100 * yr, n=1000)
            t_cool.append(t2 / day)
            L_gaia.append(float(np.interp(gaia_cadence, t, lum)) / L_sun)
            L_lsst.append(float(np.interp(lsst_cadence, t, lum)) / L_sun)

        for i in tqdm(mass_impact_parameter_indexes):
            filename = get_filename(i, 4)

            phot = photosphere(filename, orbital_period=period)

            L0.append(phot.L_phot / L_sun)
            total_mass.append(phot.snapshot.total_mass / M_earth)
            total_AM.append(phot.snapshot.total_angular_momentum)
            index.append(i)

            t, lum, R, T, t2, t10 = phot.cool(100 * yr, n=1000)
            t_cool.append(t2 / day)
            L_gaia.append(float(np.interp(gaia_cadence, t, lum)) / L_sun)
            L_lsst.append(float(np.interp(lsst_cadence, t, lum)) / L_sun)

        for i in tqdm(mass_mass_ratio_indexes):
            filename = get_filename(i, 4)

            phot = photosphere(filename, orbital_period=period)

            L0.append(phot.L_phot / L_sun)
            total_mass.append(phot.snapshot.total_mass / M_earth)
            total_AM.append(phot.snapshot.total_angular_momentum)
            index.append(i)

            t, lum, R, T, t2, t10 = phot.cool(100 * yr, n=1000)
            t_cool.append(t2 / day)
            L_gaia.append(float(np.interp(gaia_cadence, t, lum)) / L_sun)
            L_lsst.append(float(np.interp(lsst_cadence, t, lum)) / L_sun)

        L0 = np.array(L0)
        t_cool = np.array(t_cool)
        total_mass = np.array(total_mass)

        return index, L0, t_cool, total_mass, total_AM, np.array(L_gaia), np.array(L_lsst)

    i_1, L0_1, t_cool_1, m_1, AM_1, L_gaia_1, L_lsst_1 = do_all_with_period(1 * day)
    df_1 = pd.DataFrame({'i': i_1, 'L': L0_1, 't': t_cool_1, 'm': m_1, 'AM': AM_1,
                         'L_gaia': L_gaia_1, 'L_lsst': L_lsst_1})
    df_1.to_csv('res_tables/period1.csv')

    i_10, L0_10, t_cool_10, m_10, AM_10, L_gaia_10, L_lsst_10 = do_all_with_period(10 * day)
    df_10 = pd.DataFrame({'i': i_10, 'L': L0_10, 't': t_cool_10, 'm': m_10, 'AM': AM_10,
                          'L_gaia': L_gaia_10, 'L_lsst': L_lsst_10})
    df_10.to_csv('res_tables/period10.csv')

    i_100, L0_100, t_cool_100, m_100, AM_100, L_gaia_100, L_lsst_100 = do_all_with_period(100 * day)
    df_100 = pd.DataFrame({'i': i_100, 'L': L0_100, 't': t_cool_100, 'm': m_100, 'AM': AM_100,
                           'L_gaia': L_gaia_100, 'L_lsst': L_lsst_100})
    df_100.to_csv('res_tables/period100.csv')


def load_to_make_plot():

    df_1 = pd.read_csv('res_tables/period1.csv')
    df_10 = pd.read_csv('res_tables/period10.csv')
    df_100 = pd.read_csv('res_tables/period100.csv')

    norm = Normalize(vmin=0, vmax=4)

    plt.figure(figsize=(13, 9))

    sc1   = plt.scatter(df_1['t'],   df_1['L'],   marker='x', c=df_1['m'],   cmap='viridis', norm=norm, label='Period = 1 day')
    sc10  = plt.scatter(df_10['t'],  df_10['L'],  marker='*', c=df_10['m'],  cmap='viridis', norm=norm, label='Period = 10 days')
    sc100 = plt.scatter(df_100['t'], df_100['L'], marker='o', c=df_100['m'], cmap='viridis', norm=norm, label='Period = 100 days')

    plt.colorbar(sc1, label=r'Total impact mass ($M_{\oplus}$)')

    # plt.axvline(x=30, color='black', linestyle='--', linewidth=1.5, label=r'$\it{Gaia}$ cadence (30 days)')
    # plt.axvline(x=3,  color='black', linestyle=':',  linewidth=1.5, label='LSST cadence (3 days)')

    plt.legend()

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Cooling time (day)')
    plt.ylabel(r'Initial luminosity ($L_{\odot}$)')

    plt.savefig('figures/big_plot_v2.png', bbox_inches='tight')
    plt.savefig('figures/big_plot_v2.pdf', bbox_inches='tight')
    plt.close()

    # -- Cadence luminosity plot --
    # Shows L0 (hollow) and L at first possible observation (filled) for each
    # simulation.  The two panels correspond to the LSST and Gaia cadences so
    # it is immediately clear how much a short-lived bright impact fades before
    # the telescope gets a chance to look.

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=300, sharey=True)
    fig.subplots_adjust(wspace=0)

    datasets = [
        (df_1,   'o', 'Period = 1 day'),
        (df_10,  's', 'Period = 10 days'),
        (df_100, '^', 'Period = 100 days'),
    ]

    for ax, (cadence_col, cadence_days, line_col, panel_title) in zip(axs, [
        ('L_lsst', 3,  'black', r'Luminosity at first LSST observation'),
        ('L_gaia', 30, 'black', r'Luminosity at first $\it{Gaia}$ observation'),
    ]):
        sc = None
        for df, marker, lbl in datasets:
            # L0 as solid opaque markers
            sc = ax.scatter(df['t'], df['L'], s=20, c=df['m'], cmap='viridis', norm=norm,
                            marker=marker, label=lbl)
            # L at cadence as same markers, faded to show dimming
            ax.scatter(df['t'], df[cadence_col], s=20, c=df['m'], cmap='viridis',
                       norm=norm, marker=marker, alpha=0.3)
            # Arrows from L0 down to L(cadence)
            for xi, y0, yc in zip(df['t'], df['L'], df[cadence_col]):
                if yc < y0 * 0.95:
                    ax.annotate('', xy=(xi, yc), xytext=(xi, y0),
                                arrowprops=dict(arrowstyle='->', color='grey',
                                                lw=0.5, alpha=0.4))

        ax.axvline(cadence_days, color=line_col, linestyle='--', linewidth=1.5,
                   label=panel_title)
        ax.set_xlabel('Cooling time (day)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([1e-5, 5])
        ax.set_title(panel_title)

    axs[0].set_ylabel(r'Luminosity ($L_{\odot}$)')

    # Combine handles from both panels: period markers (from axs[0]) + both cadence lines
    handles0, labels0 = axs[0].get_legend_handles_labels()
    handles1, labels1 = axs[1].get_legend_handles_labels()
    # axs[0] has: period markers + LSST cadence line
    # axs[1] has: period markers (duplicates) + Gaia cadence line
    # Keep period markers once, then append the Gaia cadence line from axs[1]
    combined_handles = handles0 + [handles1[-1]]
    combined_labels  = labels0  + [labels1[-1]]
    axs[1].legend(combined_handles, combined_labels, fontsize=8, loc='upper right')

    fig.colorbar(sc, ax=axs, label=r'Total impact mass ($M_{\oplus}$)')

    plt.savefig('figures/luminosity_at_cadence.png', bbox_inches='tight')
    plt.savefig('figures/luminosity_at_cadence.pdf', bbox_inches='tight')
    plt.close()

    _table_sets = [
        ('A', range(0, 8)),
        ('B', range(8, 10)),
        ('C', range(10, 15)),
        ('D', range(15, 21)),
        ('E', range(21, 25)),
    ]
    _mr = m_impactor / (m_target + m_impactor)
    _mt = m_target + m_impactor

    # --- Table 1: initial parameters (tab:simulation_params) ---
    print(r'    \begin{tabular}{cc|ccccc}')
    print(r'        \multicolumn{2}{c|}{Simulation} & \multicolumn{5}{c}{Initial Parameters}\\')
    print(r'        Set & \# & No.\ of particles & $M_{\mathrm{total}}$ / $M_{\oplus}$ & '
          r'$\gamma$ & $b$ & $v_{\mathrm{imp}}$ / $v_{\mathrm{esc}}$ \\ \hline')
    for _set_label, _sims in _table_sets:
        _first = True
        for _i in _sims:
            _sc = _set_label if _first else ''
            print(f'        {_sc} & {_i} & {n_particles[_i]:.0f} & {_mt[_i]:.2f} & '
                  f'{_mr[_i]:.2f} & {impact_parameter[_i]:.2f} & {v_over_v_esc[_i]:.1f} \\\\')
            _first = False
    print(r'    \end{tabular}')

    # --- Table 2: model results (tab:model_results) ---
    def _build_lut(df):
        d = {}
        for _, row in df.iterrows():
            k = int(row['i'])
            if k not in d:
                d[k] = row
        return d

    lut1   = _build_lut(df_1)
    lut10  = _build_lut(df_10)
    lut100 = _build_lut(df_100)

    def _L(row): return f'{row["L"] * 1e3:.2f}'
    def _t(row): return f'{round(row["t"]):d}'

    print(r'    \begin{tabular}{cc|cc|cc|cc|cc}')
    print(r'        \multicolumn{2}{c|}{Simulation} & \multicolumn{2}{c|}{} & '
          r'\multicolumn{2}{c|}{$P = 1$ day} & \multicolumn{2}{c|}{$P = 10$ days} & '
          r'\multicolumn{2}{c}{$P = 100$ days} \\')
    print(r'        Set & \# & $M_{\mathrm{final}}$ / $M_{\oplus}$ & '
          r'$AM_{\mathrm{final}}$ / $10^{34}$ $\text{kg}\,\text{m}^2\text{s}^{-1}$ & '
          r'$L_{0}$ / $10^{-3}L_{\odot}$ & $t_{\mathrm{cool}}$ (days) & '
          r'$L_{0}$ / $10^{-3}L_{\odot}$ & $t_{\mathrm{cool}}$ (days) & '
          r'$L_{0}$ / $10^{-3}L_{\odot}$ & $t_{\mathrm{cool}}$ (days) \\ \hline')
    for _set_label, _sims in _table_sets:
        _first = True
        for _i in _sims:
            _r1, _r10, _r100 = lut1[_i], lut10[_i], lut100[_i]
            _sc = _set_label if _first else ''
            print(f'        {_sc} & {_i} & {_r10["m"]:.2f} & {_r10["AM"] / 1e34:.2f}'
                  f' & {_L(_r1)} & {_t(_r1)}'
                  f' & {_L(_r10)} & {_t(_r10)}'
                  f' & {_L(_r100)} & {_t(_r100)} \\\\')
            _first = False
    print(r'    \end{tabular}')


def test_hill_sphere():

    periods = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    # periods = [1, 10, 100]

    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.subplots_adjust(hspace=0)
    fig.set_figwidth(6.4)
    fig.set_figheight(10)

    impact_indexes = [2, 0, 5, 7]

    for i in range(len(impact_indexes)):

        L0 = []
        t_cool = []

        for T in periods:

            filename = get_filename(impact_indexes[i], 4)

            phot = photosphere(filename, orbital_period=T * day)

            L0.append(phot.L_phot / L_sun)

            t, lum, R, Tp, t_half, t_tenth = phot.cool(100 * yr, n=1000)
            t_cool.append(t_half / day)

        total_mass = m_target[impact_indexes[i]] + m_impactor[impact_indexes[i]]
        label1 = f'Sim {impact_indexes[i]} ({total_mass:.2f} $M_{{\\oplus}}$)'
        colour = viridis(i / len(impact_indexes))

        ax[0].scatter(periods, L0, marker='o', c=colour, label=label1)
        ax[1].scatter(periods, t_cool, marker='o', c=colour, label=label1)

    ax[0].set_yscale('log')
    ax[0].set_ylabel(r'Initial Luminosity ($L_{\odot}$)')

    ax[0].set_xscale('log')

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    ax[1].set_xlabel('Orbital period (days)')
    ax[1].set_ylabel('Cooling time (days)')
    ax[0].legend()

    # P = np.logspace(0, 2)
    # L = 0.0002 * P
    # T = 2000 / P

    # ax[0].plot(P, L, 'k--')
    # ax[1].plot(P, T, 'k--')

    plt.savefig('figures/hill_plot.png', bbox_inches='tight')
    plt.savefig('figures/hill_plot.pdf', bbox_inches='tight')
    plt.close()

    # --- hill_plot_v2: L0 vs t_cool for all sims ---
    # periods_v2 = [3, 30, 300, 3000]
    # markers_v2 = ['o', 'x', '+', 'D']

    # all_indexes = mass_indexes

    # fig2, ax2 = plt.subplots()
    # fig2.set_figwidth(6.4)
    # fig2.set_figheight(5)

    # prop_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # for idx, sim_i in enumerate(all_indexes):
    #     colour = prop_colours[idx % len(prop_colours)]
    #     filename = get_filename(sim_i, 4)

    #     for p_idx, T in enumerate(periods_v2):
    #         phot = photosphere(filename, orbital_period=T * day)
    #         L0 = phot.L_phot / L_sun
    #         t, lum, R, Tp, t_half, t_tenth = phot.cool(100 * yr, n=1000)
    #         ax2.scatter(t_half / day, L0, marker=markers_v2[p_idx], color=colour)

    # for p_idx, T in enumerate(periods_v2):
    #     ax2.scatter([], [], marker=markers_v2[p_idx], color='C0', label=f'Period = {T} days')

    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.set_xlabel('Cooling time (day)')
    # ax2.set_ylabel(r'Initial luminosity ($L_{\odot}$)')
    # ax2.legend()

    # plt.savefig('figures/hill_plot_v2.png', bbox_inches='tight')
    # plt.savefig('figures/hill_plot_v2.pdf', bbox_inches='tight')
    # plt.close()


def test_droplets():

    impact_indexes = list(dict.fromkeys(
        mass_indexes + mass_ratio_indexes + impact_parameter_indexes + mass_impact_parameter_indexes + mass_mass_ratio_indexes
    ))
    total_mass = (m_target + m_impactor)[impact_indexes]

    fig, ax = plt.subplots()
    plt.subplots_adjust(hspace=0)
    fig.set_figwidth(6.4)
    fig.set_figheight(5)

    L0, L0_no_infall = [], []
    t_cool, t_cool_no_infall = [], []

    for i in range(len(impact_indexes)):

        filename = get_filename(impact_indexes[i], 4)

        phot = photosphere(filename, orbital_period=orbital_period, remove_droplets=True)
        L0.append(phot.L_phot / L_sun)
        t, lum, R, Tp, t_half, t_tenth = phot.cool(100 * yr, n=1000)
        t_cool.append(t_half / day)

        phot = photosphere(filename, orbital_period=orbital_period, remove_droplets=False)
        L0_no_infall.append(phot.L_phot / L_sun)
        t, lum, R, Tp, t_half, t_tenth = phot.cool(100 * yr, n=1000)
        t_cool_no_infall.append(t_half / day)

    scatter1 = ax.scatter(t_cool, L0, c=total_mass, marker='o', label='With droplet removal')
    ax.scatter(t_cool_no_infall, L0_no_infall, c=total_mass, marker='x', label='Without droplet removal')
    ax.legend()

    plt.colorbar(scatter1, label=r'Total impact mass ($M_{\oplus}$)')

    ax.set_xlabel('Cooling time (day)')
    ax.set_ylabel(r'Initial luminosity ($L_{\odot}$)')

    ax.set_yscale('log')
    ax.set_xscale('log')

    plt.savefig('figures/droplet_plot.png', bbox_inches='tight')
    plt.savefig('figures/droplet_plot.pdf', bbox_inches='tight')
    plt.close()


def test_pressure_shell():

    impact_indexes = [2, 0, 5, 7]
    impact_labels = ['0.5', '1.0', '2.0', '4.0']

    pressures = [1e15, 1e13, 1e11, 1e9, 1e7]

    fig, ax = plt.subplots()
    plt.subplots_adjust(hspace=0)
    fig.set_figwidth(6.4)
    fig.set_figheight(5)

    for i in range(len(impact_indexes)):

        L0 = []
        t_cool = []

        for P in pressures:

            filename = get_filename(impact_indexes[i], 4)

            phot = photosphere(filename, orbital_period=orbital_period, pressure_floor=P)

            L0.append(phot.L_phot / L_sun)

            t, lum, R, Tp, t_half, t_tenth = phot.cool(100 * yr, n=1000)
            t_cool.append(t_half / day)

        label1 = f'Sim {impact_indexes[i]} ({impact_labels[i]} $M_{{\\oplus}}$)'
        colour = viridis(i / len(impact_indexes))

        ax.scatter(pressures, t_cool, marker='o', c=colour, label=label1)

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlabel('Inner pressure limit (Pa)')
    ax.set_ylabel('Cooling time (days)')
    ax.legend()

    # P = np.logspace(0, 2)
    # L = 0.0002 * P
    # T = 200 / P
    #
    # ax[0].plot(P, L, 'k--')
    # ax[1].plot(P, T, 'k--')

    plt.savefig('figures/pressure_plot.png', bbox_inches='tight')
    plt.savefig('figures/pressure_plot.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    # phase_diagram(get_filename(4, 4), sim_index=4)
    # impact_plots()
    # compute_and_save_results()
    # load_to_make_plot()
    test_hill_sphere()
    # test_pressure_shell()
    # test_droplets()
