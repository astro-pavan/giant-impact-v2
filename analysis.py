# main analysis code

import numpy as np
import matplotlib.pyplot as plt

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
           'impact parameter/impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.10_spin0.0/'
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

m_target = np.array([0.4992, 0.0999, 0.2499, 0.9995, 1.9989, 0.4992, 0.4992, 0.4992, 0.4992, 0.4992, 0.4992]) * Mearth
m_impactor = np.array([0.4992, 0.0999, 0.2499, 0.9995, 1.9989, 0.02496, 0.0249, 0.09984, 0.2496, 0.4992, 0.4992, 0.4992]) * Mearth
v_impact_z = np.array([7023, 4102, 5574, 8857, 11165, 5874, 6005, 6399, 6999, 7007, 7073])
v_impact_x = np.array([973, 566, 771, 1229, 1551, 890, 861, 893, 181, 556, 1828])
v_impact = np.sqrt(v_impact_x ** 2 + v_impact_z ** 2)
impact_parameter = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.3, 0.8]

Q_prime = modified_specific_impact_energy(m_target, m_impactor, v_impact, impact_parameter)


def luminosity_plots():
    pass


def impact_plots():
    pass


def example_extrapolation():
    pass


def phase_diagram():
    pass
