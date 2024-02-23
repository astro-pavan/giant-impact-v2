# analyses the observability of post impact bodies with Gaia

import gdr3bcg.bcg as bcg
correction_table = bcg.BolometryTable()
from astroquery.gaia import Gaia
from astropy.table import Table
from unyt import Rearth

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from snapshot_analysis import snapshot
from photosphere import photosphere, L_sun, yr
from impact_analysis import get_filename

impact_luminosity = 0.01  # L_sun
impact_temp = 2800  # K

M_bol_sun = 4.74

M_bol_impact = M_bol_sun - 2.5 * np.log10(impact_luminosity)
bol_correction_factor = correction_table.computeBc([impact_temp, 0.3, 0, 0])
M_g_impact = M_bol_impact - bol_correction_factor

print(f'Absolute magnitude (G-band) of post impact body: {M_g_impact:.2f}')


def abs_mag(g_band_mag, parallax):
    return g_band_mag + 5 * (np.log10(parallax) + 1)


def gaia_epoch_photometry():

    query = "SELECT TOP 4000 gaia_source.source_id,gaia_source.ra,gaia_source.dec,gaia_source.parallax,gaia_source.parallax_error,gaia_source.parallax_over_error,gaia_source.ruwe,gaia_source.phot_g_n_obs,gaia_source.phot_g_mean_flux,gaia_source.phot_g_mean_flux_error,gaia_source.phot_g_mean_flux_over_error,gaia_source.phot_g_mean_mag,gaia_source.phot_bp_n_obs,gaia_source.phot_bp_mean_flux_error,gaia_source.phot_bp_mean_flux_over_error,gaia_source.phot_rp_n_obs,gaia_source.phot_rp_mean_flux_error,gaia_source.phot_rp_mean_flux_over_error,gaia_source.bp_rp,gaia_source.radial_velocity,gaia_source.phot_variable_flag,gaia_source.non_single_star,gaia_source.has_xp_continuous,gaia_source.has_epoch_photometry,gaia_source.has_mcmc_gspphot,gaia_source.has_mcmc_msc,gaia_source.teff_gspphot,gaia_source.teff_gspphot_lower,gaia_source.teff_gspphot_upper,gaia_source.logg_gspphot,gaia_source.mh_gspphot,gaia_source.distance_gspphot,gaia_source.azero_gspphot,gaia_source.ag_gspphot,gaia_source.ebpminrp_gspphot\n" +\
            "FROM gaiadr3.gaia_source\n" +\
            "WHERE has_epoch_photometry = 'True'"

    print('Querying Gaia DR3 for epoch photometry...')
    job = Gaia.launch_job_async(query)
    results = job.get_results()
    print(f'Table size (rows): {len(results)}')

    retrieval_type, data_structure, data_release = 'EPOCH_PHOTOMETRY', 'INDIVIDUAL', 'Gaia DR3'

    print('Loading epoch photometry...')
    datalink = Gaia.load_data(ids=results['source_id'], data_release=data_release, retrieval_type=retrieval_type,
                              data_structure=data_structure, verbose=False, output_file=None)
    dl_keys = [inp for inp in datalink.keys()]
    dl_keys.sort()
    print('Epoch photometry loaded')

    mag, mean, single = [], [], []

    apparent_magnitude = np.array([])
    flux_fractional_error = np.array([])

    for dl_key in dl_keys:
        data_table = datalink[dl_key][0].to_table()
        data_table = Table(data_table)

        source_id = data_table['source_id'][0]

        entry = results[results['source_id'] == source_id]
        g_mag = entry['phot_g_mean_mag'][0]
        mean_flux_frac_error = 1 / entry['phot_g_mean_flux_over_error'][0]

        G_band_filter = data_table['band'] == 'G'

        single_flux_frac_error = 1 / np.mean(data_table['flux_over_error'][G_band_filter])

        mag.append(g_mag)
        mean.append(mean_flux_frac_error)
        single.append(single_flux_frac_error)

        apparent_magnitude = np.concatenate((apparent_magnitude, np.array(data_table['mag'][G_band_filter])))
        flux_fractional_error = np.concatenate((flux_fractional_error, np.array(1 / data_table['flux_over_error'][G_band_filter])))

    mag = np.array(mag)
    single = np.array(single)

    def model(x, e1, b, m1):
        return e1 * 10 ** (b * (x - m1))

    mag = apparent_magnitude
    single = flux_fractional_error

    fit = curve_fit(model, mag[mag > 16], single[mag > 16], p0=(1e-3, 0.25, 14), method='trf')
    e1, b, m1 = fit[0][0], fit[0][1], fit[0][2]
    error = lambda m: model(m, e1, b, m1)
    # fit = linregress(mag, np.log10(single))
    # m, c = fit.slope, fit.intercept
    # error = lambda x: 10 ** (m * x + c)

    error = lambda x: 10 ** np.maximum(0.25 * (x - 20) - 1.6, -3.2)

    x = np.linspace(10, 22)
    y = error(x)

    plt.hist2d(mag, np.log10(single), bins=100, cmap='viridis', rasterized=True)
    plt.plot(x, np.log10(y), 'r--')
    plt.xlabel('Apparent magnitude (G-band)')
    plt.ylabel('$\log_{10}$[Single flux mean fractional error]')
    plt.gca().invert_xaxis()
    plt.colorbar(label='Number of observations')

    plt.savefig('figures/single_flux_errors.png', bbox_inches='tight')
    plt.savefig('figures/single_flux_errors.pdf', bbox_inches='tight')
    plt.close()

    return error


def gaia_mean_flux_error_plot():

    print('Loading Gaia sample...')
    gaia_data = pd.read_csv('gaia_data.csv')
    print('Sample loaded')

    gaia_data = gaia_data[gaia_data['phot_g_mean_mag'].notna()]

    plt.hist2d(gaia_data['phot_g_mean_mag'], - np.log10(gaia_data['phot_g_mean_flux_over_error']), bins=200, cmap='viridis', rasterized=True)
    plt.xlim([14, 22])
    plt.gca().invert_xaxis()
    plt.xlabel('Apparent magnitude (G-band)')
    plt.ylabel('$\log_{10}$[Mean flux fractional error]')
    plt.savefig('figures/mean_flux_errors.png', bbox_inches='tight')
    plt.savefig('figures/mean_flux_errors.pdf', bbox_inches='tight')
    plt.close()


def gaia_analysis():

    single_flux_frac_error = gaia_epoch_photometry()
    # single_flux_frac_error = lambda x: 1e-2

    print('Loading Gaia sample...')
    gaia_data = pd.read_csv('gaia_data.csv')
    print(gaia_data.keys())
    initial_n = len(gaia_data)
    print(f'{initial_n} stars loaded')

    gaia_data = gaia_data[gaia_data['phot_g_mean_mag'].notna()]
    gaia_data = gaia_data[gaia_data['parallax'].notna()]
    gaia_data['parallax'] = np.abs(gaia_data['parallax']) * 1e-3
    gaia_data['distance'] = 1 / gaia_data['parallax']

    gaia_data['abs_g_mag'] = abs_mag(gaia_data['phot_g_mean_mag'], gaia_data['parallax'])
    gaia_data['abs_g_mag_error'] = (5 / np.log(10)) * (1 / gaia_data['parallax_over_error'])
    gaia_data['abs_g_mag_lower'] = gaia_data['abs_g_mag'] - gaia_data['abs_g_mag_error']
    gaia_data['abs_g_mag_upper'] = gaia_data['abs_g_mag'] + gaia_data['abs_g_mag_error']

    mask = (gaia_data['abs_g_mag'] < 20) & (gaia_data['abs_g_mag'] > -20) # & (gaia_data['parallax_over_error'] > 0.1)
    gaia_data = gaia_data[mask]

    filtered_n = len(gaia_data)
    print(f'Stars kept after filtering: {filtered_n / initial_n:.2%}')

    gaia_data['impact_flux_frac_difference'] = 10 ** ((gaia_data['abs_g_mag'] - M_g_impact) / 2.5)
    gaia_data['single_flux_frac_error'] = single_flux_frac_error(gaia_data['phot_g_mean_mag'])
    gaia_data['mean_flux_frac_error'] = 1 / gaia_data['phot_g_mean_flux_over_error']

    single_visible_mask = gaia_data['impact_flux_frac_difference'] > 5 * gaia_data['single_flux_frac_error']
    mean_visible_mask = gaia_data['impact_flux_frac_difference'] > 5 * gaia_data['mean_flux_frac_error']
    single_visible_frac = len(gaia_data[single_visible_mask]) / len(gaia_data)
    mean_visible_frac = len(gaia_data[mean_visible_mask]) / len(gaia_data)

    print(f'Can detect impacts around {single_visible_frac:.1%} of sample (5-sigma detection) using single measurement error')
    print(f'Can detect impacts around {mean_visible_frac:.1%} of sample (5-sigma detection) using mean measurement error')

    single_visible_mask = gaia_data['impact_flux_frac_difference'] > 3 * gaia_data['single_flux_frac_error']
    mean_visible_mask = gaia_data['impact_flux_frac_difference'] > 3 * gaia_data['mean_flux_frac_error']
    single_visible_frac = len(gaia_data[single_visible_mask]) / len(gaia_data)
    mean_visible_frac = len(gaia_data[mean_visible_mask]) / len(gaia_data)

    print(f'Can detect impacts around {single_visible_frac:.1%} of sample (3-sigma detection) using single measurement error')
    print(f'Can detect impacts around {mean_visible_frac:.1%} of sample (3-sigma detection) using mean measurement error')

    plt.hist2d(np.log10(gaia_data['single_flux_frac_error']), np.log10(gaia_data['impact_flux_frac_difference']),
               bins=100, cmap='viridis', rasterized=True, density=False)
    plt.xlabel('$\log_{10}$[Flux fractional error]')
    plt.ylabel('$\log_{10}$[Fractional change in flux from impact]')
    x = np.logspace(-4, 0)
    y3 = 3 * x
    y5 = 5 * x
    plt.plot(np.log10(x), np.log10(y3), 'r-.', label='3-sigma detection')
    plt.plot(np.log10(x), np.log10(y5), 'r--', label='5-sigma detection')
    plt.legend(loc='upper left')
    plt.colorbar(label='Number of stars')
    plt.ylim([-6, 1])
    #plt.xlim([-4, np.nanmax(np.log10(gaia_data['single_flux_frac_error']))])
    plt.savefig('figures/single_detectability.png', bbox_inches='tight')
    plt.savefig('figures/single_detectability.pdf', bbox_inches='tight')
    plt.close()

    plt.hist2d(np.log10(gaia_data['mean_flux_frac_error']), np.log10(gaia_data['impact_flux_frac_difference']),
               bins=200, cmap='viridis', rasterized=True, density=True)
    plt.xlabel('$\log_{10}$[Flux fractional error]')
    plt.ylabel('$\log_{10}$[Fractional change in flux from impact]')
    x = np.logspace(-4, 0)
    y3 = 3 * x
    y5 = 5 * x
    plt.plot(np.log10(x), np.log10(y3), 'r-.', label='3-sigma detection')
    plt.plot(np.log10(x), np.log10(y5), 'r--', label='5-sigma detection')
    plt.legend(loc='upper left')
    plt.ylim([-6, 1])
    plt.xlim([-4, np.nanmax(np.log10(gaia_data['mean_flux_frac_error']))])
    plt.colorbar(label='Star density')
    plt.savefig('figures/mean_detectability.png', bbox_inches='tight')
    plt.savefig('figures/mean_detectability.pdf', bbox_inches='tight')
    plt.close()

    visible_data = gaia_data[single_visible_mask]

    plt.hist(visible_data['phot_g_mean_mag'], density=True, bins=400, histtype='step', label='Stars with potentially observable impacts')
    plt.hist(gaia_data['phot_g_mean_mag'], density=True, bins=400, histtype='step', label='All gaia stars')
    plt.xlabel('Apparent magnitude (G-band)')
    plt.legend()
    plt.savefig('figures/mag_detectability.png', bbox_inches='tight')
    plt.savefig('figures/mag_detectability.pdf', bbox_inches='tight')
    plt.close()

    plt.hist(visible_data['bp_rp'], density=True, bins=400, histtype='step', label='Stars with potentially observable impacts')
    plt.xlabel('BP-RP colour')
    plt.legend()
    plt.xlim([-1, 6])
    plt.savefig('figures/bp_rp_detectability.png', bbox_inches='tight')
    plt.savefig('figures/bp_rp_detectability.pdf', bbox_inches='tight')
    plt.close()

    plt.hist(visible_data['abs_g_mag'], density=True, bins=400, histtype='step',
             label='Stars with potentially observable impacts')
    plt.hist(gaia_data['abs_g_mag'], density=True, bins=400, histtype='step', label='All gaia stars')
    plt.xlabel('Absolute magnitude (G-band)')
    plt.legend()
    plt.xlim([0, 20])
    plt.savefig('figures/abs_mag_detectability.png', bbox_inches='tight')
    plt.savefig('figures/abs_mag_detectability.pdf', bbox_inches='tight')
    plt.close()

    nan_mask = visible_data['bp_rp'].notna() &\
               visible_data['abs_g_mag'].notna() &\
               (visible_data['parallax_over_error'] > 5) &\
               visible_data['teff_gspphot'].notna() &\
               visible_data['logg_gspphot'].notna() &\
               visible_data['mh_gspphot'].notna()

    good_stars = visible_data[nan_mask]

    plt.hist(good_stars['phot_g_mean_mag'], density=True, bins=200, histtype='step',
             label='Stars with potentially observable impacts')
    plt.hist(gaia_data['phot_g_mean_mag'][gaia_data['parallax_over_error'] > 10], density=True, bins=200, histtype='step',
             label='All gaia stars')
    plt.xlabel('Apparent magnitude (G-band)')
    plt.legend()
    plt.xlim([0, 20])
    plt.savefig('figures/mag_detectability_good.png', bbox_inches='tight')
    plt.savefig('figures/mag_detectability_good.pdf', bbox_inches='tight')
    plt.close()

    plt.hist(good_stars['abs_g_mag'], density=True, bins=200, histtype='step',
             label='Stars with potentially observable impacts')
    plt.hist(gaia_data['abs_g_mag'][gaia_data['parallax_over_error'] > 10], density=True, bins=200, histtype='step', label='All gaia stars')
    plt.xlabel('Absolute magnitude (G-band)')
    plt.legend()
    plt.xlim([0, 20])
    plt.savefig('figures/abs_mag_detectability_good.png', bbox_inches='tight')
    plt.savefig('figures/abs_mag_detectability_good.pdf', bbox_inches='tight')
    plt.close()

    plt.hist2d(good_stars['teff_gspphot'], good_stars['abs_g_mag'], bins=100, cmap='inferno', rasterized=True)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Absolute magnitude (G-band)')
    plt.xlim([np.nanmin(good_stars['teff_gspphot']), 5000])
    plt.axvline(3700, 0, 1, color='white', linestyle='--')
    plt.annotate(text='K', xy=(0.25, 0.1), c='white', xycoords='axes fraction')
    plt.annotate(text='M', xy=(0.75, 0.1), c='white', xycoords='axes fraction')
    plt.colorbar(label='Number of stars')
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.savefig('figures/detectability_HR.png', bbox_inches='tight')
    plt.savefig('figures/detectability_HR.pdf', bbox_inches='tight')
    plt.close()

    return good_stars


def gaia_analysis_v2():

    single_flux_frac_error = gaia_epoch_photometry()

    print('Loading Gaia sample...')
    gaia_data = pd.read_csv('gaia_data.csv')
    print(gaia_data.keys())
    initial_n = len(gaia_data)
    print(f'{initial_n} stars loaded')

    gaia_data = gaia_data[gaia_data['phot_g_mean_mag'].notna()]
    gaia_data = gaia_data[gaia_data['parallax'].notna()]
    gaia_data['parallax'] = np.abs(gaia_data['parallax']) * 1e-3

    gaia_data['abs_g_mag'] = abs_mag(gaia_data['phot_g_mean_mag'], gaia_data['parallax'])

    mask = (gaia_data['abs_g_mag'] < 20) & (gaia_data['abs_g_mag'] > -20) # & (gaia_data['parallax_over_error'] > 0.1)
    gaia_data = gaia_data[mask]

    filtered_n = len(gaia_data)
    print(f'Stars kept after filtering: {filtered_n / initial_n:.2%}')

    gaia_data['single_flux_frac_error'] = single_flux_frac_error(gaia_data['phot_g_mean_mag'])

    delta_F_over_F = gaia_data['single_flux_frac_error']
    m_g_star = gaia_data['phot_g_mean_mag']
    p = gaia_data['parallax']

    delta_m = 2.5 * np.log10(1 + delta_F_over_F)

    m_g_I_3_sigma = m_g_star - 2.5 * np.log10((10 ** ((3 * delta_m) / 2.5)) - 1)
    m_g_I_5_sigma = m_g_star - 2.5 * np.log10((10 ** ((5 * delta_m) / 2.5)) - 1)

    M_g_I_3_sigma = m_g_I_3_sigma + (5 * (np.log10(p) + 1))
    M_g_I_5_sigma = m_g_I_5_sigma + (5 * (np.log10(p) + 1))

    M_bol_I_3_sigma = M_g_I_3_sigma + bol_correction_factor
    M_bol_I_5_sigma = M_g_I_5_sigma + bol_correction_factor

    L_I_min_3_sigma = 10 ** (0.4 * (M_bol_sun - M_bol_I_3_sigma))
    L_I_min_5_sigma = 10 ** (0.4 * (M_bol_sun - M_bol_I_5_sigma))

    detectable_mask = L_I_min_3_sigma < 5e-3
    frac = len(L_I_min_3_sigma[detectable_mask]) / len(gaia_data)

    print(f'Detectable fraction: {frac:.2%}')

    plt.hist(np.log10(L_I_min_3_sigma), bins=500, density=True, cumulative=True, range=(-4, 3), log=True,
             histtype='step', label='3-sigma detection')
    plt.hist(np.log10(L_I_min_5_sigma), bins=500, density=True, cumulative=True, range=(-4, 3), log=True,
             histtype='step', label='5-sigma detection')
    plt.legend(loc='lower right')
    plt.xlabel('Minimum observable impact luminosity ($L_{\odot}$)')
    plt.ylabel('Fraction of stars')

    xlabels = ['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$']
    xlabel_pos = [-3, -2, -1, 0, 1]
    plt.xticks(xlabel_pos, xlabels)

    plt.xlim([-3.5, 1])
    plt.ylim([1e-3, 1e0])

    plt.grid(which='both', axis='y')
    plt.grid(which='major', axis='x')

    plt.savefig('figures/cumlulative_hist.png', bbox_inches='tight')
    plt.savefig('figures/cumlulative_hist.pdf', bbox_inches='tight')
    plt.close()

    potential_stars = gaia_data[detectable_mask]

    nan_mask = potential_stars['bp_rp'].notna() & \
               potential_stars['abs_g_mag'].notna() & \
               (potential_stars['parallax_over_error'] > 5) & \
               potential_stars['teff_gspphot'].notna() & \
               potential_stars['logg_gspphot'].notna() & \
               potential_stars['mh_gspphot'].notna()

    good_stars = potential_stars[nan_mask]

    plt.hist2d(good_stars['teff_gspphot'], good_stars['abs_g_mag'], bins=100, cmap='inferno', rasterized=True)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Absolute magnitude (G-band)')
    plt.xlim([np.nanmin(good_stars['teff_gspphot']), 5000])
    plt.axvline(3700, 0, 1, color='white', linestyle='--')
    plt.annotate(text='K', xy=(0.25, 0.1), c='white', xycoords='axes fraction')
    plt.annotate(text='M', xy=(0.75, 0.1), c='white', xycoords='axes fraction')
    plt.colorbar(label='Number of stars')
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.savefig('figures/detectability_HR_v2.png', bbox_inches='tight')
    plt.savefig('figures/detectability_HR_v2.pdf', bbox_inches='tight')
    plt.close()


def simulated_light_curve(gaia_entry):

    gaia_entry = gaia_entry[gaia_entry['phot_g_mean_mag'] > 18.5]
    #gaia_entry = gaia_entry[gaia_entry['teff_gspphot'] > 4500]

    m_g = np.array(gaia_entry['phot_g_mean_mag'])[0]
    parallax = np.array(gaia_entry['parallax'])[0]
    T_eff = np.array(gaia_entry['teff_gspphot'])[0]
    log_g = np.array(gaia_entry['logg_gspphot'])[0]
    fe_h = np.array(gaia_entry['mh_gspphot'])[0]
    alpha_h = 0
    bcf = correction_table.computeBc([T_eff, log_g, fe_h, alpha_h])
    print(f'Bolometric correction factor: {bcf:.4f}')

    M_g = m_g + 5 * (np.log10(parallax) + 1)
    M_bol = M_g + bcf
    L = 10 ** ((M_bol_sun - M_bol) / 2.5)

    print(f'm_g = {m_g:.2f}')
    print(f'L = {L:.2e} L_sun')
    print(f'T = {T_eff} K')
    print(f'log_g = {log_g}')
    print(f'd = {1/parallax}')

    filename = get_filename(0, 4)
    snap = snapshot(filename)

    phot = photosphere(snap, 12 * Rearth, 60 * Rearth, 800, n_theta=20)
    phot.set_up()
    L0 = phot.luminosity / L_sun
    time, lum, A, R, T, m_dot, t2, t10 = phot.long_term_evolution()

    lum = lum / L_sun
    time = time / yr

    time = np.concatenate([[-0.0001], time])
    lum = np.concatenate([[0], lum])

    light_curve_2 = interp1d(time, lum, bounds_error=False, fill_value=0)

    filename = get_filename(15, 4)
    snap = snapshot(filename)

    phot = photosphere(snap, 12 * Rearth, 60 * Rearth, 800, n_theta=20)
    phot.set_up()
    L0 = phot.luminosity / L_sun
    time, lum, A, R, T, m_dot, t2, t10 = phot.long_term_evolution()

    lum = lum / L_sun
    time = time / yr

    time = np.concatenate([[-0.0001], time])
    lum = np.concatenate([[0], lum])

    light_curve_3 = interp1d(time, lum, bounds_error=False, fill_value=0)

    t_sample = (np.arange(-15, 70) * (30/365)) + (15/365)
    t_continuous = np.linspace(-1, 5, num=3000)

    L_2 = L + light_curve_2(t_sample)
    L_3 = L + light_curve_3(t_sample)

    L_2_model = L + light_curve_2(t_continuous)
    L_3_model = L + light_curve_3(t_continuous)

    L_base = np.full_like(t_sample, L)

    frac_error = np.array(gaia_entry['single_flux_frac_error'])[0]

    print(f'Impact luminosity: {L0:.2e} L_sun')
    print(f'Fractional error: {frac_error:.1e}')

    L_2 = np.random.normal(L_2, L_2 * frac_error)
    L_3 = np.random.normal(L_3, L_3 * frac_error)
    L_base = np.random.normal(L_base, L_base * frac_error)

    m_g_2 = M_bol_sun - 2.5 * np.log10(L_2) - bcf - 5 * (np.log10(parallax) + 1)
    m_g_3 = M_bol_sun - 2.5 * np.log10(L_3) - bcf - 5 * (np.log10(parallax) + 1)
    m_g_base = M_bol_sun - 2.5 * np.log10(L_base) - bcf - 5 * (np.log10(parallax) + 1)

    m_g_2_model = M_bol_sun - 2.5 * np.log10(L_2_model) - bcf - 5 * (np.log10(parallax) + 1)
    m_g_3_model = M_bol_sun - 2.5 * np.log10(L_3_model) - bcf - 5 * (np.log10(parallax) + 1)

    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.set_figwidth(16)
    fig.set_figheight(12)
    plt.subplots_adjust(hspace=0)

    xlim = [-0.5, 3]

    axs[0].scatter(t_sample, m_g_2, color='blue', s=10)
    axs[0].errorbar(t_sample, m_g_2, yerr=frac_error, fmt='none', color='blue')
    axs[0].plot(t_continuous, m_g_2_model, 'b--')
    #axs[0].set_ylim([m_g - 0.12, m_g + 0.01])
    axs[0].invert_yaxis()
    axs[0].set_xlim(xlim)
    axs[0].axhline(m_g, 0, 1, color='black', linestyle='--')

    axs[1].scatter(t_sample, m_g_3, color='blue', s=10)
    axs[1].errorbar(t_sample, m_g_3, yerr=frac_error, fmt='none', color='blue')
    axs[1].plot(t_continuous, m_g_3_model, 'b--')
    #axs[1].set_ylim([m_g - 0.12, m_g + 0.01])
    axs[1].invert_yaxis()
    axs[1].set_ylabel('Apparent magnitude (Gaia G-band)')
    axs[1].set_xlim(xlim)
    axs[1].axhline(m_g, 0, 1, color='black', linestyle='--')

    axs[2].scatter(t_sample, m_g_base, color='blue', s=0.5)
    axs[2].errorbar(t_sample, m_g_base, yerr=frac_error, fmt='none', color='blue')
    #axs[2].set_ylim([m_g - 0.12, m_g + 0.01])
    axs[2].invert_yaxis()
    axs[2].set_xlabel('Time (yr)')
    axs[2].set_xlim(xlim)
    axs[2].axhline(m_g, 0, 1, color='black', linestyle='--')

    plt.savefig('figures/gaia_light_curve.png', bbox_inches='tight')
    plt.savefig('figures/gaia_light_curve.pdf', bbox_inches='tight')

    fig.set_figwidth(8)
    fig.set_figheight(5)

    axs[0].set_xlim([-0.5, 3])
    axs[1].set_xlim([-0.5, 3])
    axs[1].set_xlabel('Time (yr)')
    # axs[2].set_xlim([-0.5, 3])

    plt.savefig('figures/gaia_light_curve_small.png', bbox_inches='tight')
    plt.savefig('figures/gaia_light_curve_small.pdf', bbox_inches='tight')


if __name__ == '__main__':
    # gaia_mean_flux_error_plot()
    # stars = gaia_analysis()
    # simulated_light_curve(stars)
    gaia_analysis_v2()


