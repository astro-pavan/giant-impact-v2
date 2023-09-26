# main analysis code

import numpy as np
import matplotlib.pyplot as plt
from unyt import Rearth
from tqdm import tqdm

from snapshot_analysis import snapshot, gas_slice, data_labels
from photosphere import photosphere, M_earth, L_sun, yr
import EOS as fst

Mearth = 5.972e24


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

subpath = ['impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.50_spin0.0/',
           'target mass/impact_p1.0e+05_M0.1_ratio1.00_v1.10_b0.50_spin0.0/',
           'target mass/impact_p1.0e+05_M0.2_ratio1.00_v1.10_b0.50_spin0.0/',
           'target mass/impact_p1.0e+05_M1.0_ratio1.00_v1.10_b0.50_spin0.0/',
           'target mass/impact_p1.0e+05_M2.0_ratio1.00_v1.10_b0.50_spin0.0/',
           'mass ratio/impact_p1.0e+05_M0.5_ratio0.05_v1.10_b0.50_spin0.0/',
           'mass ratio/impact_p1.0e+05_M0.5_ratio0.20_v1.10_b0.50_spin0.0/',
           'mass ratio/impact_p1.0e+05_M0.5_ratio0.50_v1.10_b0.50_spin0.0/',
           'impact parameter/impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.10_spin0.0/',
           'impact parameter/impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.30_spin0.0/',
           'impact parameter/impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.80_spin0.0/']

snapshot_names = [
    'output/snapshot_0003.hdf5',
    'output/snapshot_0006.hdf5',
    'output/snapshot_0015.hdf5',
    'output/snapshot_0051.hdf5',
    'output/snapshot_0240.hdf5'
]

snapshot_times = [0, 0.5, 2, 8, 40]

m_target = np.array([0.5,
                     0.1, 0.25, 1.0, 2,
                     0.5, 0.5, 0.5,
                     0.5, 0.5, 0.5]) * Mearth

m_impactor = np.array([0.5,
                       0.1, 0.25, 1.0, 2.0,
                       0.025, 0.1, 0.25,
                       0.5, 0.5, 0.5]) * Mearth

v_over_v_esc = np.array([1.1,
                         1.1, 1.1, 1.1, 1.1,
                         1.1, 1.1, 1.1,
                         1.1, 1.1, 1.1])

v_impact_z = np.array([7023,
                       4102, 5574, 8857, 11165,
                       5874, 6005, 6399,
                       6999, 7007, 7073])

v_impact_x = np.array([973,
                       566, 771, 1229, 1551,
                       890, 861, 893,
                       181, 556, 1828])

impact_parameter = np.array([0.5,
                             0.5, 0.5, 0.5, 0.5,
                             0.5, 0.5, 0.5,
                             0.1, 0.3, 0.8])

v_impact = np.sqrt(v_impact_x ** 2 + v_impact_z ** 2)
Q_prime = modified_specific_impact_energy(m_target, m_impactor, v_impact, impact_parameter)

indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
missing_or_errors = [10]


def get_filename(i, i_time):
    return f'{snapshot_path}{subpath[i]}{snapshot_names[i_time]}'


def luminosity_plots():

    L = np.zeros_like(Q_prime)
    AM = np.zeros_like(Q_prime)
    SAM = np.zeros_like(Q_prime)

    light_curve = []
    time = []

    for i in [0, 1, 2, 3, 4]:
        filename = get_filename(i, 4)
        snap = snapshot(filename)

        phot = photosphere(snap, 12 * Rearth, 50 * Rearth, 400, n_theta=20)
        phot.set_up()
        # phot.plot('alpha')
        L[i] = phot.luminosity / L_sun
        t, lum, A, R, T, m = phot.long_term_evolution()
        light_curve.append(lum)
        time.append(t)
        AM[i] = snap.total_angular_momentum
        SAM[i] = snap.total_specific_angular_momentum


    # target mass
    change_mass_indexes = [0, 1, 2, 3, 4]
    for i in change_mass_indexes:
        plt.plot(time[i] / yr, light_curve[i] / L_sun, label=f'Target mass = {m_target[i] / Mearth:.2f}' + '$M_{\oplus}$')
    plt.xlabel('Time (yr)')
    plt.ylabel('Luminosity ($L_{\odot}$)')
    plt.legend()
    plt.show()

    # plt.scatter(Q_prime, L)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Modified specific Impact Energy (J/kg)')
    # plt.ylabel('Luminosity ($L_{\odot}$)')
    #
    # plt.savefig('figures/QL_plot.png', bbox_inches='tight')
    # plt.savefig('figures/QL_plot.pdf', bbox_inches='tight')
    # plt.close()


def single_analysis(i):

    filename = get_filename(i, 4)
    snap = snapshot(filename)
    # s = gas_slice(snap, size=8)
    # s.full_plot()
    phot = photosphere(snap, 12 * Rearth, 60 * Rearth, 500, n_theta=40)

    # phot.plot('rho', plot_photosphere=True)
    # phot.plot('T', log=False, round_to=1000, plot_photosphere=True, val_max=8000)
    # phot.plot('P', plot_photosphere=True)
    # phot.plot('s', log=False, round_to=1000, plot_photosphere=True)

    phot.set_up()

    # phot.plot('rho', plot_photosphere=True)
    # phot.plot('T', log=False, round_to=1000, plot_photosphere=True, val_max=8000)
    #phot.plot('P', plot_photosphere=True)
    # phot.plot('s', log=False, round_to=1000, plot_photosphere=True)
    # phot.plot('tau', plot_photosphere=True)

    time, lum, A, R, T, m = phot.long_term_evolution(plot=False, plot_interval=100)
    plt.plot(time / yr, lum / L_sun)
    plt.show()

    plt.plot(time / yr, R)
    plt.show()

    plt.plot(time / yr, A)
    plt.show()


def impact_plots():
    # default impact
    # change b (up and down?)
    # change gamma
    # change m (up and down?)
    # 4-6 impacts shown

    sims = [0, 8, 6, 4]
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
            if j != 0:
                ax[i, j].set_ylabel('')
                ax[i, j].set_yticklabels([])
            if i == 0:
                ax[i, j].set_title(f'T = {snapshot_times[j]:.1f} hrs')

    plt.subplots_adjust(wspace=0.02)
    plt.subplots_adjust(hspace=0.02)

    plt.savefig('figures/impact_plot.png', bbox_inches='tight')
    plt.savefig('figures/impact_plot.pdf', bbox_inches='tight')
    plt.close()


def example_extrapolation():
    pass


def phase_diagram(filename):
    S = np.linspace(2500, 12000, num=100)
    P = np.logspace(1, 9, num=100)
    x, y = np.meshgrid(S, P)
    rho, T = fst.rho_EOS(x, y), fst.T1_EOS(x, y)
    z = np.log10(fst.alpha(rho, T, y, x))
    z = np.where(np.isfinite(z), z, np.NaN)
    plt.figure(figsize=(13, 9))
    plt.contourf(x, y, z, 200, cmap='viridis')
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
    plt.annotate('Liquid', (4200, 1e8), c='white')
    plt.annotate('Liquid + Vapour', (6300, 1e4), c='white')
    plt.annotate('Vapour', (9800, 1e8), c='white')

    snap = snapshot(filename)
    phot = photosphere(snap, 12 * Rearth, 35 * Rearth, 600, n_theta=40)

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


def make_table():
    for i in range(len(m_target)):
        mass_ratio = m_impactor / m_target
        print(f'{m_target[i]/Mearth:.1f} & {impact_parameter[i]:.1f} & {mass_ratio[i]:.2f} & {v_over_v_esc[i]:.1f} & LUM \\\\')


single_analysis(0)

# 0 is unstable with HSE (problem with entropy extrapolation) and long term evolution

# 1 has problems with long term evolution
# 2 has an issue with late stage time evolution
# 3 has some odd spikes in evolution
# 4 has similar spikes

# 5 has problems with long term evolution (its so dim it may not even be worth it)
# 6 has weird evolution
# 7 has problems with long term evolution

# 8 has issues with evolution
# 9 has problems with long term evolution
# 10 breaks at entropy extrapolation
