import gdr3bcg.bcg as bcg
correction_table = bcg.BolometryTable()
from astroquery.gaia import Gaia
from astropy.table import Table
from unyt import Rearth

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from snapshot_analysis import snapshot, gas_slice, data_labels
from photosphere import photosphere, M_earth, L_sun, yr
from impact_analysis import get_filename

impact_luminosity = 5e-3  # L_sun
impact_luminosity_2 = 1e-3  # L_sun
impact_temp = 2800  # K

M_bol_sun = 4.74

M_bol_impact = M_bol_sun - 2.5 * np.log10(impact_luminosity)
M_bol_impact_2 = M_bol_sun - 2.5 * np.log10(impact_luminosity_2)
bol_correction_factor = correction_table.computeBc([impact_temp, 0.3, 0, 0])
M_g_impact = M_bol_impact - bol_correction_factor
M_g_impact_2 = M_bol_impact_2 - bol_correction_factor

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

    mag = np.array(mag)
    single = np.array(single)

    def model(x, e1, b, m1):
        return e1 * 10 ** (b * (x - m1))

    fit = curve_fit(model, mag[mag > 16], single[mag > 16], p0=(1e-3, 0.25, 14))
    e1, b, m1 = fit[0][0], fit[0][1], fit[0][2]
    error = lambda m: model(m, e1, b, m1)
    # fit = linregress(mag, np.log10(single))
    # m, c = fit.slope, fit.intercept
    # error = lambda x: 10 ** (m * x + c)

    x = np.linspace(10, 22)
    y = error(x)

    plt.hist2d(mag, np.log10(single), bins=200, cmap='viridis', rasterized=True)
    plt.plot(x, np.log10(y), 'r--')
    plt.xlabel('Apparent magnitude (G-band)')
    plt.ylabel('$\log_{10}$[Single flux mean fractional error]')
    plt.gca().invert_xaxis()
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
    print('Sample loaded')
    print(gaia_data.keys())
    initial_n = len(gaia_data)

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
               bins=200, cmap='viridis', rasterized=True)
    plt.xlabel('$\log_{10}$[Flux fractional error]')
    plt.ylabel('$\log_{10}$[Fractional change in flux from impact]')
    x = np.logspace(-4, 0)
    y3 = 3 * x
    y5 = 5 * x
    plt.plot(np.log10(x), np.log10(y3), 'r-.', label='3-$\sigma$ detection')
    plt.plot(np.log10(x), np.log10(y5), 'r--', label='5-$\sigma$ detection')
    plt.legend(loc='upper left')
    plt.ylim([-6, 1])
    plt.xlim([-4, np.nanmax(np.log10(gaia_data['single_flux_frac_error']))])
    plt.savefig('figures/single_detectability.png', bbox_inches='tight')
    plt.savefig('figures/single_detectability.pdf', bbox_inches='tight')
    plt.close()

    plt.hist2d(np.log10(gaia_data['mean_flux_frac_error']), np.log10(gaia_data['impact_flux_frac_difference']),
               bins=200, cmap='viridis', rasterized=True)
    plt.xlabel('$\log_{10}$[Flux fractional error]')
    plt.ylabel('$\log_{10}$[Fractional change in flux from impact]')
    x = np.logspace(-4, 0)
    y3 = 3 * x
    y5 = 5 * x
    plt.plot(np.log10(x), np.log10(y3), 'r-.', label='3-$\sigma$ detection')
    plt.plot(np.log10(x), np.log10(y5), 'r--', label='5-$\sigma$ detection')
    plt.legend(loc='upper left')
    plt.ylim([-6, 1])
    plt.xlim([-4, np.nanmax(np.log10(gaia_data['mean_flux_frac_error']))])
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

    nan_mask = visible_data['bp_rp'].notna() &\
               visible_data['abs_g_mag'].notna() &\
               (visible_data['parallax_over_error'] > 10) &\
               visible_data['teff_gspphot'].notna() &\
               visible_data['logg_gspphot'].notna() &\
               visible_data['mh_gspphot'].notna()

    good_stars = visible_data[nan_mask]

    return good_stars


def simulated_light_curve(gaia_entry):

    gaia_entry = gaia_entry[gaia_entry['phot_g_mean_mag'] > 17]

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
    print(f'p = {parallax}')

    filename = get_filename(0, 4)
    snap = snapshot(filename)

    phot = photosphere(snap, 12 * Rearth, 60 * Rearth, 400, n_theta=10)
    phot.set_up()
    L0 = phot.luminosity / L_sun
    time, lum, A, R, T, m_dot, t2, t4, t10 = phot.long_term_evolution_v2()

    lum = lum / L_sun
    time = time / yr

    time = np.concatenate([[-0.0001], time])
    lum = np.concatenate([[0], lum])

    light_curve = interp1d(time, lum, bounds_error=False, fill_value=0)

    t_sample = (np.arange(-15, 70) * (30/365)) + (15/365)
    L_total = L + light_curve(t_sample)
    frac_error = np.array(gaia_entry['single_flux_frac_error'])[0]
    print(f'Impact luminosity: {L0:.2e} L_sun')
    print(f'Fractional error: {frac_error:.1e}')
    L_total = np.random.normal(L_total, L_total * frac_error)

    M_bol_2 = M_bol_sun - 2.5 * np.log10(L_total)
    M_g_2 = M_bol_2 - bcf
    m_g_2 = M_g_2 - 5 * (np.log10(parallax) + 1)

    plt.scatter(t_sample, m_g_2)
    plt.errorbar(t_sample, m_g_2, yerr=frac_error, fmt='none')
    plt.xlabel('Time (yr)')
    plt.ylabel('Apparent magnitude (G-band)')
    plt.xlim([-1, 4])
    plt.axhline(m_g, 0, 1, color='black', linestyle='--')
    plt.gca().invert_yaxis()
    plt.savefig('figures/gaia_light_curve.png', bbox_inches='tight')
    plt.savefig('figures/gaia_light_curve.pdf', bbox_inches='tight')


if __name__ == '__main__':
    gaia_mean_flux_error_plot()
    stars = gaia_analysis()
    simulated_light_curve(stars)



