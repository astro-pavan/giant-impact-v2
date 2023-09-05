# main analysis code

import numpy as np
import matplotlib.pyplot as plt
from unyt import Rearth
from tqdm import tqdm

from snapshot_analysis import snapshot, gas_slice, data_labels
from photosphere import photosphere
import forsterite2 as fst

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

m_target = np.array([0.4992,
                     0.0999, 0.2499, 0.9995, 1.9989,
                     0.4992, 0.4992, 0.4992,
                     0.4992, 0.4992, 0.4992]) * Mearth

m_impactor = np.array([0.4992,
                       0.0999, 0.2499, 0.9995, 1.9989,
                       0.0250, 0.0998, 0.2496,
                       0.4992, 0.4992, 0.4992]) * Mearth

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

indexes = [0, 1, 3, 4, 5, 6, 8, 9]
missing_or_errors = [2, 7, 10]


def get_filename(i, i_time):
    return f'{snapshot_path}{subpath[i]}{snapshot_names[i_time]}'


def luminosity_plots():

    L = np.zeros_like(Q_prime)
    AM = np.zeros_like(Q_prime)
    SAM = np.zeros_like(Q_prime)

    for i in indexes:
        filename = get_filename(i, 4)
        snap = snapshot(filename)
        phot = photosphere(snap, 12 * Rearth, 35 * Rearth, 600, n_theta=40)
        phot.analyse(plot_check=True)
        L[i] = phot.L_phot
        AM[i] = snap.total_angular_momentum
        SAM[i] = snap.total_specific_angular_momentum

    plt.scatter(Q_prime, L / 3.8e26)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Modified specific Impact Energy (J/kg)')
    plt.ylabel('Luminosity ($L_{\odot}$)')
    plt.show()


def impact_plots():
    pass


def example_extrapolation():
    pass


def phase_diagram(filename):
    S = np.linspace(2500, 12000, num=100)
    P = np.logspace(1, 9, num=100)
    x, y = np.meshgrid(S, P)
    rho, T = fst.rho_EOS(x, y), fst.T1_EOS(x, y)
    z = np.log10(fst.alpha(rho, T, y, x))
    plt.contourf(x, y, z, 200, cmap='plasma')
    cbar = plt.colorbar(label='$\log_{10}$[' + data_labels['alpha'] + ']')

    tick_positions = np.arange(np.ceil(np.nanmin(z)), np.ceil(np.nanmax(z)), 2)
    cbar.set_ticks(tick_positions)

    plt.yscale('log')
    plt.xlabel(data_labels['s'])
    plt.ylabel(data_labels['P'])
    plt.plot(fst.NewEOS.vc.Sl, fst.NewEOS.vc.Pl, 'w-', label='Vapor Dome')
    plt.plot(fst.NewEOS.vc.Sv, fst.NewEOS.vc.Pv, 'w-')
    plt.scatter(fst.S_critical_point, fst.P_critical_point, c='white', label='Critical Point')
    plt.legend(loc='lower left')
    plt.xlim([4000, 12000])
    plt.ylim([1e1, 1e9])
    plt.annotate('L', (4400, 1e8), c='white')
    plt.annotate('L+V', (6500, 1e4), c='white')
    plt.annotate('V', (10000, 1e8), c='white')

    snap = snapshot(filename)
    phot = photosphere(snap, 12 * Rearth, 35 * Rearth, 600, n_theta=40)
    #phot.analyse()

    S, P = phot.data['s'][20, :], phot.data['P'][20, :]
    plt.plot(S, P, 'k--')

    rad = np.array([2, 4, 6, 8, 10, 12, 16, 20, 30])
    r_labels = []
    S_points, P_points = np.zeros_like(rad), np.zeros_like(rad)
    for i in range(len(rad)):
        r_labels.append(f'{rad[i]}' + ' $R_{\oplus}$')
        j = phot.get_index(rad[i] * 6371000, 0)[1]
        S_points[i], P_points[i] = phot.data['s'][20, j], phot.data['P'][20, j]

    plt.scatter(S_points, P_points, color='black', s=8, marker='o')
    for j in range(len(rad)):
        plt.annotate(r_labels[j], (S_points[j], P_points[j]), xytext=(-37, -5), textcoords='offset points', color='black')

    plt.show()


phase_diagram(filename=get_filename(0, 4))
