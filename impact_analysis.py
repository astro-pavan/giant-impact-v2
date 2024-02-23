# main analysis code
import matplotlib.colors
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
orbital_period = 20 * day
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

snapshot_path = '/home/pavan/Project/Final_Sims/'

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
Q_prime = modified_specific_impact_energy(m_target * M_earth, m_impactor* M_earth, v_impact, impact_parameter)

mass_indexes = [2, 3, 0, 4, 5, 6, 7]

impact_parameter_indexes = [15, 16, 17, 18, 0, 19]
mass_ratio_indexes = [8, 9, 10, 0]

mass_mass_ratio_indexes = [11, 12, 10, 14]
mass_impact_parameter_indexes = [23, 15, 24, 25]


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
    plt.ylabel('Luminosity ($L_{\odot}$)')
    plt.legend(title='Total mass ($M_{\oplus}$)')

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
    labels = ['$M_{target}$ = 0.5 $M_{\oplus}$\n$\gamma$ = 0.50\nb = 0.5',
              '$M_{target}$ $\\rightarrow$ 0.25 $M_{\oplus}$',
              '$M_{target}$ $\\rightarrow$ 1.0 $M_{\oplus}$',
              '$\gamma$ $\\rightarrow$ 0.33',
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


# produces a phase diagram with the thermal profile of a post impact body
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
    phot = photosphere(snap, 12 * Rearth, resolution=2000, period=100*day, n_theta=40)

    S, P = phot.data['s'][20, :], phot.data['P'][20, :]
    plt.plot(S, P, color='black', linestyle='--', label='Thermal profile with droplets', zorder=5)

    # rad = np.array([2, 4, 6, 8, 10, 15, 20, 25, 30])
    # r_labels = []
    # S_points, P_points = np.zeros_like(rad), np.zeros_like(rad)
    # for i in range(len(rad)):
    #     r_labels.append(f'{rad[i]}' + ' $R_{\oplus}$')
    #     j = phot.get_index(rad[i] * 6371000, 0)[1]
    #     S_points[i], P_points[i] = phot.data['s'][20, j], phot.data['P'][20, j]
    #
    # plt.scatter(S_points, P_points, color='black', s=8, marker='o', zorder=4)
    # for j in range(len(rad)):
    #     plt.annotate(r_labels[j], (S_points[j], P_points[j]), xytext=(-37, -5), textcoords='offset points',
    #                  color='black', zorder=5)

    phot.initial_cool(1e5)
    phot.get_photosphere()

    # tau = phot.data['tau'][20, :]
    # j = np.argmax(tau < 2/3)
    # print(phot.data['R'][20, j] / 6371000)
    # S_phot, P_phot = phot.data['s'][20, j], phot.data['P'][20, j]
    # plt.scatter(S_phot, P_phot, marker='x', color='red', label='$\\tau$ = $\\frac{2}{3}$', zorder=6)

    # plt.arrow(7000, 4e6, 650, 0, color='red', width=10, head_width=40, head_length=30, length_includes_head=True)

    phot.remove_droplets()

    S, P = phot.data['s'][20, :], phot.data['P'][20, :]
    plt.plot(S, P, color='red', linestyle='-.', label='Thermal profile without droplets')

    for i in range(10):
        phot.cool_step(0.01 * yr)
        phot.remove_droplets(dt=0.01 * yr)

    S, P = phot.data['s'][20, :], phot.data['P'][20, :]
    plt.plot(S, P, color='purple', linestyle='-.', label='Thermal profile after cooling')

    for i in range(90):
        phot.cool_step(0.01 * yr)
        phot.remove_droplets(dt=0.01 * yr)

    S, P = phot.data['s'][20, :], phot.data['P'][20, :]
    plt.plot(S, P, color='magenta', linestyle='-.', label='Thermal profile after cooling')

    plt.legend(loc='lower left')
    plt.savefig('figures/phase_diagram.png', bbox_inches='tight')
    plt.savefig('figures/phase_diagram.pdf', bbox_inches='tight')
    plt.close()
    phot.plot('P', plot_photosphere=True, val_min=1, val_max=1e12)


# produces a large plot that summarizes the simulation results
def full_plot():
    L0_m, L0_b, L0_mb, L0_mr = [], [], [], []
    t_cool_m, t_cool_b, t_cool_mb, t_cool_mr = [], [], [], []

    L0 = np.zeros_like(m_target)
    t_cool = np.zeros_like(m_target)

    for i in mass_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

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

        phot = photosphere(snap, 12 * Rearth, resolution=res, period=orbital_period, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0_b.append(phot.luminosity / L_sun)
        L0[i] = phot.luminosity / L_sun

        t, lum, A, R, T, m_dot, t2, t10 = phot.long_term_evolution()
        t_cool_b.append(t2)
        t_cool[i] = t2

    for i in mass_impact_parameter_indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

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

        phot = photosphere(snap, 12 * Rearth, resolution=res, period=orbital_period, n_theta=n_theta, n_phi=n_phi)
        phot.set_up()
        L0_mr.append(phot.luminosity / L_sun)
        L0[i] = phot.luminosity / L_sun

        t, lum, A, R, T, m_dot, t2, t10 = phot.long_term_evolution()
        t_cool_mr.append(t2)
        t_cool[i] = t2

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

    axs[0, 0].scatter(total_mass[mass_indexes], L0_m, c='#E69F00', label='b = 0.5, $\gamma$ = 0.50')
    axs[0, 0].scatter(total_mass[mass_impact_parameter_indexes], L0_mb, c='#56B4E9', label='b = 0.1, $\gamma$ = 0.50')
    axs[0, 0].scatter(total_mass[mass_mass_ratio_indexes], L0_mr, c='#009E73', label='b = 0.5, $\gamma$ = 0.33')

    axs[0, 0].plot()

    axs[1, 0].scatter(total_mass[mass_indexes], t_cool_m / day, c='#E69F00', label='b = 0.5, $\gamma$ = 0.50')
    axs[1, 0].scatter(total_mass[mass_impact_parameter_indexes], t_cool_mb / day, c='#56B4E9', label='b = 0.1, $\gamma$ = 0.50')
    axs[1, 0].scatter(total_mass[mass_mass_ratio_indexes], t_cool_mr / day, c='#009E73', label='b = 0.5, $\gamma$ = 0.33')

    axs[0, 1].scatter(impact_parameter[impact_parameter_indexes], L0_b, c='#CC79A7', label='$M_{\mathrm{total}}$ = 1.0 $M_{\oplus}$, $\gamma$ = 0.50')

    axs[1, 1].scatter(impact_parameter[impact_parameter_indexes], t_cool_b / day, c='#CC79A7')

    axs[0, 0].legend()
    axs[0, 1].legend()

    axs[0, 0].set_ylim([0, 0.04])
    axs[0, 1].set_ylim([0, 0.04])
    axs[1, 0].set_ylim([10, 450])
    axs[1, 1].set_ylim([10, 450])

    axs[1, 0].set_yscale('log')
    axs[1, 1].set_yscale('log')

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

    axs[0, 0].annotate(text='A', xy=(0.95, 0.05), xycoords='axes fraction')
    axs[0, 1].annotate(text='B', xy=(0.95, 0.05), xycoords='axes fraction')
    axs[1, 0].annotate(text='C', xy=(0.95, 0.05), xycoords='axes fraction')
    axs[1, 1].annotate(text='D', xy=(0.95, 0.05), xycoords='axes fraction')

    plt.savefig('figures/big_plot.png', bbox_inches='tight')
    plt.savefig('figures/big_plot.pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.style.use('dark_background')

    size_factor = 40

    mass_ratio_mask = np.full_like(L0, False, dtype=bool)
    mass_ratio_mask[mass_mass_ratio_indexes] = True

    scatter1 = plt.scatter(t_cool[~mass_ratio_mask] / day, L0[~mass_ratio_mask], s=total_mass[~mass_ratio_mask] * size_factor,
                c=impact_parameter[~mass_ratio_mask], cmap='viridis', marker='o')
    # plt.scatter(t_cool[mass_ratio_mask] / day, L0[mass_ratio_mask], s=total_mass[mass_ratio_mask] * size_factor,
    #             c=impact_parameter[mass_ratio_mask], cmap='spring', marker='^')

    plt.colorbar(scatter1, label='Impact parameter (b)')
    plt.clim(0, 0.6)

    plt.xlim([0, (np.max(t_cool) / day) + 50])

    sizes = np.array([0.25, 0.5, 1, 2, 4])
    legend_handles = [plt.scatter([], [], s=size * size_factor, label=f'{size:.2f}', color='white', marker='o') for size in sizes]
    plt.legend(handles=legend_handles, title='Total Mass ($M_{\oplus}$)')

    plt.xlabel('Cooling time (day)')
    plt.ylabel('Initial luminosity ($L_{\odot}$)')

    plt.yscale('log')

    plt.savefig('figures/big_plot_v2.png', bbox_inches='tight')
    plt.savefig('figures/big_plot_v2.pdf', bbox_inches='tight')
    plt.close()

    plt.style.use('default')


def test_hill_sphere():

    periods = [1, 2, 5, 10, 20, 50, 100]

    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.subplots_adjust(hspace=0)
    fig.set_figwidth(6.4)
    fig.set_figheight(10)

    impact_indexes = [2, 0, 5, 7]
    impact_labels = ['0.5', '1.0', '2.0', '4.0']

    for i in range(len(impact_indexes)):

        L0, L0_no_infall = [], []
        t_cool, t_cool_no_infall = [], []

        for T in periods:

            filename = get_filename(impact_indexes[i], 4)
            snap = snapshot(filename)

            phot = photosphere(snap, 12 * Rearth, resolution=res, period=T*day, n_theta=n_theta, n_phi=n_phi)
            phot_no_infall = photosphere(snap, 12 * Rearth, resolution=res, period=T*day, n_theta=n_theta, n_phi=n_phi, droplet_infall=False)

            phot.set_up()
            phot_no_infall.set_up()

            L0.append(phot.luminosity / L_sun)
            L0_no_infall.append(phot.luminosity / L_sun)

            t, lum, A, R, T, m_dot, t2, t10 = phot.long_term_evolution()
            t_cool.append(t2 / day)
            t, lum, A, R, T, m_dot, t2, t10 = phot_no_infall.long_term_evolution()
            t_cool_no_infall.append(t2 / day)

        label1 = impact_labels[i]
        label2 = '(no droplet infall)'
        colour = viridis(i / len(impact_indexes))

        ax[0].scatter(periods, L0, marker='o', c=colour, label=label1)
        #ax[0].scatter(periods, L0, marker='x', c=colour, label=label2)

        ax[1].scatter(periods, t_cool, marker='o', c=colour, label=label1)
        #ax[1].scatter(periods, t_cool_no_infall, marker='x', c=colour, label=label2)

    ax[0].set_yscale('log')

    #plt.xlabel('Orbital period (days)')
    ax[0].set_ylabel('Initial Luminosity ($L_{\odot}$)')

    ax[0].set_xscale('log')

    # plt.savefig('figures/hill_L_plot.png', bbox_inches='tight')
    # plt.savefig('figures/hill_L_plot.pdf', bbox_inches='tight')
    # plt.close()

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    ax[1].set_xlabel('Orbital period (days)')
    ax[1].set_ylabel('Cooling time (days)')
    ax[0].legend(title='Total mass ($M_{\oplus}$)')
    # plt.legend()

    plt.savefig('figures/hill_plot.png', bbox_inches='tight')
    plt.savefig('figures/hill_plot.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    # example_extrapolation()

    # generate_table()
    # phase_diagram(get_filename(0, 4))
    # impact_plots()
    # light_curves(mass_indexes)
    full_plot()
    # test_hill_sphere()
