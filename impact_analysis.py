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


def get_filename(i, i_time):
    return f'{snapshot_path}{subpath[i]}{snapshot_names[i_time]}'


def light_curves():

    indexes = [0, 1, 2, 3, 4, 7, 8, 9]

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

        phot = photosphere(snap, 12 * Rearth, 60 * Rearth, 400, n_theta=20)
        phot.set_up()
        L0.append(phot.luminosity / L_sun)
        t, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2(save_name=f'impact{i}', plot=False, plot_interval=0.1)
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
        plt.plot(time[i] / yr, light_curve[i] / L_sun, label=f'Total mass = {(m_target[i] + m_impactor[i]) / Mearth:.2f}' + '$M_{\oplus}$')
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

    plt.xlim([0.01, 60])
    plt.xscale('log')
    plt.savefig('figures/mass_light_curve_log.pdf', bbox_inches='tight')
    plt.savefig('figures/mass_light_curve_log.png', bbox_inches='tight')

    plt.close()

    total_mass = (m_target + m_impactor)[indexes]

    plt.scatter(total_mass[change_mass_indexes] / Mearth, t_half[change_mass_indexes] / yr, label='$t_{1/2}$')
    plt.scatter(total_mass[change_mass_indexes] / Mearth, t_quarter[change_mass_indexes] / yr, label='$t_{1/4}$')
    plt.scatter(total_mass[change_mass_indexes] / Mearth, t_tenth[change_mass_indexes] / yr, label='$t_{1/10}$')
    plt.xlabel('Total mass ($M_{\oplus}$)')
    plt.ylabel('Time (yr)')

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
        print(f'{m_target[i] / Mearth:.1f} &'
              f' {impact_parameter[i]:.1f} &'
              f' {mass_ratio[i]:.2f} &'
              f' {v_over_v_esc[i]:.1f} &'
              f' {L0[i] / L_sun:.1e} &'
              f' {t_half[i] / yr:.2f}  \\\\')


def luminosity_plots():
    pass


def single_analysis(i):

    filename = get_filename(i, 4)
    snap = snapshot(filename)
    s = gas_slice(snap, size=20)
    s.full_plot()
    phot = photosphere(snap, 12 * Rearth, 60 * Rearth, 500, n_theta=40)

    # phot.plot('rho', plot_photosphere=True)
    # phot.plot('T', log=False, round_to=1000, plot_photosphere=True, val_max=8000)
    # phot.plot('P', plot_photosphere=True)
    # phot.plot('s', log=False, round_to=1000, plot_photosphere=True)

    phot.set_up()

    phot.plot('rho', plot_photosphere=True)
    # phot.plot('T', log=False, round_to=1000, plot_photosphere=True, val_max=8000)
    # phot.plot('P', plot_photosphere=True)
    # phot.plot('s', log=False, round_to=1000, plot_photosphere=True)
    phot.plot('tau', plot_photosphere=True)

    time, lum, A, R, T, m, t_half, t_tenth = phot.long_term_evolution(plot=True, plot_interval=100)
    plt.plot(time / yr, lum / L_sun)
    plt.show()

    plt.plot(time / yr, R)
    plt.show()

    plt.plot(time / yr, A)
    plt.show()

    plt.plot(time / yr, T)
    plt.show()


def impact_plots():

    sims = [0, 2, 3, 6, 8]
    labels = ['$M_{target}$ = 0.5 $M_{\oplus}$\n$\gamma$ = 1\nb = 0.5',
              '$M_{target}$ $\\rightarrow$ 0.25 $M_{\oplus}$',
              '$M_{target}$ $\\rightarrow$ 1.0 $M_{\oplus}$',
              '$\gamma$ $\\rightarrow$ 0.5',
              'b $\\rightarrow$ 0.1']
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
                ax[i, j].annotate(labels[i], (0.1, 0.7) if i == 0 else (0.1, 0.9), xycoords='axes fraction', color='white')
            if i == 0:
                ax[i, j].set_title(f'T = {snapshot_times[j]:.1f} hrs')

    plt.subplots_adjust(wspace=0.02)
    plt.subplots_adjust(hspace=0.02)

    plt.savefig('figures/impact_plot.png', bbox_inches='tight')
    plt.savefig('figures/impact_plot.pdf', bbox_inches='tight')
    plt.close()


def example_extrapolation():

    filename = get_filename(3, 4)
    snap = snapshot(filename)
    s = gas_slice(snap, size=18)
    R, theta = snap.HD_limit_R.value / 6371000, np.linspace(0, 2*np.pi, num=1000)
    curve = [R*np.cos(theta), R*np.sin(theta)]
    s.plot('rho', threshold=1e-6, save='density_floor', show=False, curve=curve, curve_label='Extrapolation point')
    phot = photosphere(snap, 12 * Rearth, 60 * Rearth, 1600, n_theta=100)
    phot.set_up()
    phot.plot('P', plot_photosphere=True, save='P_extrapolation', val_min=100, val_max=1e10, ylim=[-25, 25], xlim=[0, 50], cmap='plasma')


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

    for i in range(len(m_target)):
        mass_ratio = m_impactor / m_target
        print(f'{m_target[i]/Mearth:.1f} & {impact_parameter[i]:.1f} & {mass_ratio[i]:.2f} & {v_over_v_esc[i]:.1f} & LUM \\\\')


light_curves()

# 5 is too dim
# 6 doesn't work
# 7 seems ok

# 8 seems ok
# 9 also works
# 10 is unbound
