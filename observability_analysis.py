# type: ignore

# analyses the observability of post impact bodies with Gaia
# import requests.exceptions

import gaiadr3_bcg.gdr3bcg.bcg as bcg
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
from photosphere import photosphere, L_sun, yr, day
from impact_analysis import get_filename, m_target, m_impactor

import re
import os

impact_luminosity = 0.01  # L_sun
impact_temp = 2800  # K

M_bol_sun = 4.66

M_bol_impact = M_bol_sun - 2.5 * np.log10(impact_luminosity)
bol_correction_factor = correction_table.computeBc([impact_temp, 0.3, 0, 0])
M_g_impact = M_bol_impact - bol_correction_factor

print(f'Absolute magnitude (G-band) of post impact body: {M_g_impact:.2f}')


def abs_mag(g_band_mag, parallax):
    return g_band_mag + 5 * (np.log10(parallax) + 1)


def gaia_epoch_photometry():

    cache_file = 'gaia_epoch_photometry_cache.npz'

    if os.path.exists(cache_file):
        print('Loading epoch photometry from local cache...')
        cache = np.load(cache_file)
        apparent_magnitude = cache['apparent_magnitude']
        flux_fractional_error = cache['flux_fractional_error']
    else:
        query = "SELECT TOP 10 gaia_source.source_id,gaia_source.ra,gaia_source.dec,gaia_source.parallax,gaia_source.parallax_error,gaia_source.parallax_over_error,gaia_source.ruwe,gaia_source.phot_g_n_obs,gaia_source.phot_g_mean_flux,gaia_source.phot_g_mean_flux_error,gaia_source.phot_g_mean_flux_over_error,gaia_source.phot_g_mean_mag,gaia_source.phot_bp_n_obs,gaia_source.phot_bp_mean_flux_error,gaia_source.phot_bp_mean_flux_over_error,gaia_source.phot_rp_n_obs,gaia_source.phot_rp_mean_flux_error,gaia_source.phot_rp_mean_flux_over_error,gaia_source.bp_rp,gaia_source.radial_velocity,gaia_source.phot_variable_flag,gaia_source.non_single_star,gaia_source.has_xp_continuous,gaia_source.has_epoch_photometry,gaia_source.has_mcmc_gspphot,gaia_source.has_mcmc_msc,gaia_source.teff_gspphot,gaia_source.teff_gspphot_lower,gaia_source.teff_gspphot_upper,gaia_source.logg_gspphot,gaia_source.mh_gspphot,gaia_source.distance_gspphot,gaia_source.azero_gspphot,gaia_source.ag_gspphot,gaia_source.ebpminrp_gspphot\n" +\
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

        apparent_magnitude = np.array([])
        flux_fractional_error = np.array([])

        for dl_key in dl_keys:
            data_table = datalink[dl_key][0].to_table()
            data_table = Table(data_table)

            apparent_magnitude = np.concatenate((apparent_magnitude, np.array(data_table['g_transit_mag'])))
            flux_fractional_error = np.concatenate((flux_fractional_error, np.array(1 / data_table['g_transit_flux_over_error'])))

        np.savez(cache_file, apparent_magnitude=apparent_magnitude, flux_fractional_error=flux_fractional_error)
        print(f'Epoch photometry saved to {cache_file}')

    def model(x, e1, b, m1):
        return e1 * 10 ** (b * (x - m1))

    mag = np.nan_to_num(apparent_magnitude)
    single = np.nan_to_num(flux_fractional_error)

    fit = curve_fit(model, mag[mag > 16], single[mag > 16], p0=(1e-3, 0.25, 14), method='trf')
    e1, b, m1 = fit[0][0], fit[0][1], fit[0][2]
    error = lambda m: model(m, e1, b, m1)
    # fit = linregress(mag, np.log10(single))
    # m, c = fit.slope, fit.intercept
    # error = lambda x: 10 ** (m * x + c)

    error = lambda x: 10 ** np.maximum(0.25 * (x - 20) - 1.6, -3.2)

    x = np.linspace(10, 22)
    y = error(x)

    # print(np.log10(single))

    inf_mask = ~np.isinf(np.log10(single))

    plt.hist2d(mag[inf_mask], np.log10(single)[inf_mask], bins=100, cmap='viridis', rasterized=True)
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
    gaia_data = pd.read_csv('GaiaSource_000000-003111.csv', header=1000)
    print(gaia_data.keys())
    initial_n = len(gaia_data)
    print(f'{initial_n} stars loaded')

    gaia_data = gaia_data[gaia_data['phot_g_mean_mag'].notna()]
    gaia_data = gaia_data[gaia_data['parallax'].notna()]
    gaia_data['parallax'] = np.abs(gaia_data['parallax']) * 1e-3

    gaia_data['abs_g_mag'] = abs_mag(gaia_data['phot_g_mean_mag'], gaia_data['parallax'])

    # mask = (gaia_data['abs_g_mag'] < 20) & (gaia_data['abs_g_mag'] > -20) # & (gaia_data['parallax_over_error'] > 0.1)
    # gaia_data = gaia_data[mask]

    # filtered_n = len(gaia_data)
    # print(f'Stars kept after filtering: {filtered_n / initial_n:.2%}')

    gaia_data['single_flux_frac_error'] = single_flux_frac_error(gaia_data['phot_g_mean_mag'])
    delta_m_LSST = np.where(gaia_data['phot_g_mean_mag'] > 16, 0.005, 10000)

    delta_F_over_F = gaia_data['single_flux_frac_error']
    m_g_star = gaia_data['phot_g_mean_mag']
    p = gaia_data['parallax']

    delta_m = 2.5 * np.log10(1 + delta_F_over_F)

    m_g_I_3_sigma = m_g_star - 2.5 * np.log10((10 ** ((3 * delta_m) / 2.5)) - 1)
    m_g_I_5_sigma = m_g_star - 2.5 * np.log10((10 ** ((5 * delta_m) / 2.5)) - 1)

    m_g_I_3_sigma_LSST = m_g_star - 2.5 * np.log10((10 ** ((3 * delta_m_LSST) / 2.5)) - 1)
    m_g_I_5_sigma_LSST = m_g_star - 2.5 * np.log10((10 ** ((5 * delta_m_LSST) / 2.5)) - 1)

    M_g_I_3_sigma = m_g_I_3_sigma + (5 * (np.log10(p) + 1))
    M_g_I_5_sigma = m_g_I_5_sigma + (5 * (np.log10(p) + 1))

    M_g_I_3_sigma_LSST = m_g_I_3_sigma_LSST + (5 * (np.log10(p) + 1))
    M_g_I_5_sigma_LSST = m_g_I_5_sigma_LSST + (5 * (np.log10(p) + 1))

    M_bol_I_3_sigma = M_g_I_3_sigma + bol_correction_factor
    M_bol_I_5_sigma = M_g_I_5_sigma + bol_correction_factor

    M_bol_I_3_sigma_LSST = M_g_I_3_sigma_LSST + bol_correction_factor
    M_bol_I_5_sigma_LSST = M_g_I_5_sigma_LSST + bol_correction_factor

    L_I_min_3_sigma = 10 ** (0.4 * (M_bol_sun - M_bol_I_3_sigma))
    L_I_min_5_sigma = 10 ** (0.4 * (M_bol_sun - M_bol_I_5_sigma))

    L_I_min_3_sigma_LSST = 10 ** (0.4 * (M_bol_sun - M_bol_I_3_sigma_LSST))
    L_I_min_5_sigma_LSST = 10 ** (0.4 * (M_bol_sun - M_bol_I_5_sigma_LSST))

    detectable_mask = L_I_min_3_sigma < 5e-3
    frac = len(L_I_min_3_sigma[detectable_mask]) / len(gaia_data)

    print(f'Detectable fraction: {frac:.2%}')

    plt.figure(dpi=300)

    n, bins, patches = plt.hist(np.log10(L_I_min_3_sigma), bins=500, density=True, cumulative=True, range=(-4, 3),
                                log=True, histtype='step', label='3-sigma detection')
    plt.hist(np.log10(L_I_min_5_sigma), bins=500, density=True, cumulative=True, range=(-4, 3),
             log=True, histtype='step', label='5-sigma detection')
    plt.legend(loc='lower right')
    plt.xlabel('Impact luminosity ($L_{\odot}$)')
    plt.ylabel('Fraction of stars where impact is detectable')

    xlabels = ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$']
    xlabel_pos = [-5, -4, -3, -2, -1, 0, 1]
    plt.xticks(xlabel_pos, xlabels)

    plt.xlim([-5, 1])
    plt.ylim([1e-3, 1e0])

    plt.savefig('figures/cumlulative_hist_no_lines.png', bbox_inches='tight')
    plt.savefig('figures/cumlulative_hist_no_lines.pdf', bbox_inches='tight')

    plt.grid(which='both', axis='y')
    plt.grid(which='major', axis='x')

    plt.savefig('figures/cumlulative_hist.png', bbox_inches='tight')
    plt.savefig('figures/cumlulative_hist.pdf', bbox_inches='tight')
    plt.close()

    plt.hist(np.log10(L_I_min_3_sigma), bins=500, density=True, cumulative=True, range=(-4, 3),
                                log=True, histtype='step', label='3-sigma detection (with Gaia)', color='blue', linestyle='-')
    plt.hist(np.log10(L_I_min_5_sigma), bins=500, density=True, cumulative=True, range=(-4, 3),
             log=True, histtype='step', label='5-sigma detection (with Gaia)', color='blue', linestyle='--')
    n_LSST, bins_LSST, patches = plt.hist(np.log10(L_I_min_3_sigma_LSST), bins=500, density=True, cumulative=True, range=(-4, 3),
                                log=True, histtype='step', label='3-sigma detection (with LSST)', color='red', linestyle='-')
    plt.hist(np.log10(L_I_min_5_sigma_LSST), bins=500, density=True, cumulative=True, range=(-4, 3),
             log=True, histtype='step', label='5-sigma detection (with LSST)', color='red', linestyle='--')
    plt.legend(loc='lower right')
    plt.xlabel('Impact luminosity ($L_{\odot}$)')
    plt.ylabel('Fraction of Gaia stars where impact is detectable')

    plt.xticks(xlabel_pos, xlabels)

    plt.xlim([-5, 1])
    plt.ylim([1e-3, 1e0])

    plt.savefig('figures/cumlulative_hist_no_lines_LSST.png', bbox_inches='tight')
    plt.savefig('figures/cumlulative_hist_no_lines_LSST.pdf', bbox_inches='tight')

    potential_stars = gaia_data[detectable_mask]

    plt.figure()

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

    observability_probability = lambda L: np.interp(np.log10(L), bins[:-1], n)
    observability_probability_LSST = lambda L: np.interp(np.log10(L), bins_LSST[:-1], n_LSST)

    return good_stars, observability_probability, observability_probability_LSST


def LSST_analysis():

    print('Loading Gaia stars within 50 pc...')
    gaia_data = pd.read_csv('stars_within_50_pc.csv')
    print(gaia_data.keys())
    initial_n = len(gaia_data)
    print(f'{initial_n} stars loaded')

    gaia_data = gaia_data[gaia_data['phot_g_mean_mag'].notna()]
    gaia_data = gaia_data[gaia_data['parallax'].notna()]
    gaia_data['parallax'] = np.abs(gaia_data['parallax']) * 1e-3

    gaia_data['abs_g_mag'] = abs_mag(gaia_data['phot_g_mean_mag'], gaia_data['parallax'])

    plt.hist(gaia_data['abs_g_mag'], bins=50)
    plt.show()

    nan_mask = gaia_data['bp_rp'].notna()
    good_stars = gaia_data[nan_mask]

    WD_mask = -2*gaia_data['bp_rp'] + good_stars['abs_g_mag'] < 7
    good_stars = good_stars[WD_mask]

    plt.hist2d(good_stars['bp_rp'], good_stars['abs_g_mag'], bins=50, cmap='inferno')
    plt.gca().invert_yaxis()
    plt.show()

    plt.hist(good_stars['abs_g_mag'], bins=50)
    plt.show()


def simulated_light_curve(gaia_entry, size=(16, 12)):

    plt.hist(gaia_entry['phot_g_mean_mag'], bins=50)
    plt.show()
    plt.close()

    # gaia_entry = gaia_entry[gaia_entry['abs_g_mag'] < 7]
    gaia_entry = gaia_entry[gaia_entry['phot_g_mean_mag'] > 18.8]

    plt.hist(gaia_entry['phot_g_mean_mag'], bins=50)
    plt.show()
    plt.close()

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
    print(f'd = {1/parallax} pc')

    i1, i2, i3 = 2, 5, 6

    def _label(i):
        mt = m_target[i] + m_impactor[i]
        return f'Simulation {i} ($M_{{\\mathrm{{total}}}}$ = {mt:.1f} $M_{{\\oplus}}$)'

    label1, label2, label3 = _label(i1), _label(i2), _label(i3)

    filename = get_filename(i1, 4)

    phot = photosphere(filename, orbital_period=100*day)
    time, lum, R, T, t_half, t_tenth = phot.cool(20 * yr, n=10000, max_dt=day)
    lum1, time1 = lum / L_sun, time / yr
    R_curve_1 = interp1d(time1, R / 6371000, bounds_error=False, fill_value=(R[0] / 6371000, R[-1] / 6371000))
    T_curve_1 = interp1d(time1, T, bounds_error=False, fill_value=(T[0], T[-1]))
    time1, lum1 = np.concatenate([[-0.0001], time1]), np.concatenate([[0], lum1])
    light_curve_1 = interp1d(time1, lum1, bounds_error=False, fill_value=0)

    filename = get_filename(i2, 4)

    phot = photosphere(filename, orbital_period=20*day)
    time, lum, R, T, t_half, t_tenth = phot.cool(100 * yr, n=10000, max_dt=day)
    lum2, time2 = lum / L_sun, time / yr
    R_curve_2 = interp1d(time2, R / 6371000, bounds_error=False, fill_value=(R[0] / 6371000, R[-1] / 6371000))
    T_curve_2 = interp1d(time2, T, bounds_error=False, fill_value=(T[0], T[-1]))
    time2, lum2 = np.concatenate([[-0.0001], time2]), np.concatenate([[0], lum2])
    light_curve_2 = interp1d(time2, lum2, bounds_error=False, fill_value=0)

    filename = get_filename(i3, 4)

    phot = photosphere(filename, orbital_period=100*day)
    time, lum, R, T, t_half, t_tenth = phot.cool(20 * yr, n=10000, max_dt=day)
    lum3, time3 = lum / L_sun, time / yr
    R_curve_3 = interp1d(time3, R / 6371000, bounds_error=False, fill_value=(R[0] / 6371000, R[-1] / 6371000))
    T_curve_3 = interp1d(time3, T, bounds_error=False, fill_value=(T[0], T[-1]))
    time3, lum3 = np.concatenate([[-0.0001], time3]), np.concatenate([[0], lum3])
    light_curve_3 = interp1d(time3, lum3, bounds_error=False, fill_value=0)

    plt.plot(time1, lum1)
    plt.plot(time2, lum2)
    plt.plot(time3, lum3)

    plt.yscale('log')

    plt.savefig('cooling.png')
    plt.close()

    t_end = 7 # yr

    t_sample = (np.arange(-15, 100) * (30/365)) + (15/365)
    t_continuous = np.linspace(-1, t_end, num=3000)

    L_1 = L + light_curve_1(t_sample)
    L_2 = L + light_curve_2(t_sample)
    L_3 = L + light_curve_3(t_sample)

    L_1_model = L + light_curve_1(t_continuous)
    L_2_model = L + light_curve_2(t_continuous)
    L_3_model = L + light_curve_3(t_continuous)

    L_base = np.full_like(t_sample, L)

    frac_error = np.array(gaia_entry['single_flux_frac_error'])[0]
    mag_error = 2.5 * np.log10(1 + frac_error)

    L_1_true, L_2_true, L_3_true = L_1.copy(), L_2.copy(), L_3.copy()

    L_1 = np.random.normal(L_1, L_1 * frac_error)
    L_2 = np.random.normal(L_2, L_2 * frac_error)
    L_3 = np.random.normal(L_3, L_3 * frac_error)
    L_base = np.random.normal(L_base, L_base * frac_error)

    m_g_1 = M_bol_sun - 2.5 * np.log10(L_1) - bcf - 5 * (np.log10(parallax) + 1)
    m_g_2 = M_bol_sun - 2.5 * np.log10(L_2) - bcf - 5 * (np.log10(parallax) + 1)
    m_g_3 = M_bol_sun - 2.5 * np.log10(L_3) - bcf - 5 * (np.log10(parallax) + 1)
    m_g_base = M_bol_sun - 2.5 * np.log10(L_base) - bcf - 5 * (np.log10(parallax) + 1)

    m_g_1_model = M_bol_sun - 2.5 * np.log10(L_1_model) - bcf - 5 * (np.log10(parallax) + 1)
    m_g_2_model = M_bol_sun - 2.5 * np.log10(L_2_model) - bcf - 5 * (np.log10(parallax) + 1)
    m_g_3_model = M_bol_sun - 2.5 * np.log10(L_3_model) - bcf - 5 * (np.log10(parallax) + 1)

    fig, ax = plt.subplots()
    fig.set_figwidth(size[0])
    fig.set_figheight(size[1] / 3)
    fig.set_dpi(300)

    xlim = [-0.5, t_end]
    colors = ['tab:blue', 'tab:orange']

    ax.scatter(t_sample, m_g_1, color=colors[0], s=10)
    ax.errorbar(t_sample, m_g_1, yerr=mag_error, fmt='none', color=colors[0])
    ax.scatter(t_sample, m_g_3, color=colors[1], s=10)
    ax.errorbar(t_sample, m_g_3, yerr=mag_error, fmt='none', color=colors[1])
    ax.invert_yaxis()
    ax.set_xlim(xlim)
    ax.axhline(m_g, 0, 1, color='black', linestyle='--')
    ax.set_ylabel('Apparent magnitude (Gaia G-band)')
    ax.set_xlabel('Time (yr)')

    plt.savefig('figures/gaia_light_curve_no_lines.png', bbox_inches='tight')
    plt.savefig('figures/gaia_light_curve_no_lines.pdf', bbox_inches='tight')

    ax.plot(t_continuous, m_g_1_model, '--', color=colors[0], label=label1)
    ax.plot(t_continuous, m_g_3_model, '--', color=colors[1], label=label3)
    ax.legend()

    plt.savefig('figures/gaia_light_curve.png', bbox_inches='tight')
    plt.savefig('figures/gaia_light_curve.pdf', bbox_inches='tight')

    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(4)
    fig.set_dpi(300)
    plt.subplots_adjust(hspace=0)

    ax.scatter(t_sample, m_g_1, color='blue', s=10)
    ax.errorbar(t_sample, m_g_1, yerr=mag_error, fmt='none', color='blue')
    ax.invert_yaxis()
    ax.set_xlim(xlim)
    ax.axhline(m_g, 0, 1, color='black', linestyle='--')
    ax.annotate(label1, (0.75, 0.9), xycoords='axes fraction')
    ax.set_ylabel('Apparent magnitude (Gaia G-band)')
    ax.set_xlabel('Time (yr)')

    plt.savefig('figures/gaia_light_curve_1.png', bbox_inches='tight')
    plt.savefig('figures/gaia_light_curve_1.pdf', bbox_inches='tight')

    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(4)
    fig.set_dpi(300)
    plt.subplots_adjust(hspace=0)

    ax.scatter(t_sample, m_g_2, color='blue', s=10)
    ax.errorbar(t_sample, m_g_2, yerr=mag_error, fmt='none', color='blue')
    ax.invert_yaxis()
    ax.set_xlim(xlim)
    ax.axhline(m_g, 0, 1, color='black', linestyle='--')
    ax.annotate(label2, (0.75, 0.9), xycoords='axes fraction')
    ax.set_ylabel('Apparent magnitude (Gaia G-band)')
    ax.set_xlabel('Time (yr)')

    plt.savefig('figures/gaia_light_curve_2.png', bbox_inches='tight')
    plt.savefig('figures/gaia_light_curve_2.pdf', bbox_inches='tight')

    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(4)
    fig.set_dpi(300)
    plt.subplots_adjust(hspace=0)

    ax.scatter(t_sample, m_g_3, color='blue', s=10)
    ax.errorbar(t_sample, m_g_3, yerr=mag_error, fmt='none', color='blue')
    ax.invert_yaxis()
    ax.set_xlim(xlim)
    ax.axhline(m_g, 0, 1, color='black', linestyle='--')
    ax.annotate(label3, (0.75, 0.9), xycoords='axes fraction')
    ax.set_ylabel('Apparent magnitude (Gaia G-band)')
    ax.set_xlabel('Time (yr)')

    plt.savefig('figures/gaia_light_curve_3.png', bbox_inches='tight')
    plt.savefig('figures/gaia_light_curve_3.pdf', bbox_inches='tight')

    # --- Luminosity versions ---

    L_lum_err_1 = L_1_true * frac_error
    L_lum_err_2 = L_2_true * frac_error
    L_lum_err_3 = L_3_true * frac_error

    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.set_figwidth(size[0])
    fig.set_figheight(size[1])
    fig.set_dpi(300)
    plt.subplots_adjust(hspace=0)

    axs[0].scatter(t_sample, L_1, color='blue', s=10)
    axs[0].errorbar(t_sample, L_1, yerr=L_lum_err_1, fmt='none', color='blue')
    axs[0].set_xlim(xlim)
    axs[0].axhline(L, 0, 1, color='black', linestyle='--')
    axs[0].annotate(label1, (0.82, 0.9), xycoords='axes fraction')
    axs[0].set_yscale('log')

    axs[1].scatter(t_sample, L_2, color='blue', s=10)
    axs[1].errorbar(t_sample, L_2, yerr=L_lum_err_2, fmt='none', color='blue')
    axs[1].set_ylabel('Luminosity ($L_{\odot}$)')
    axs[1].set_xlim(xlim)
    axs[1].axhline(L, 0, 1, color='black', linestyle='--')
    axs[1].annotate(label2, (0.82, 0.9), xycoords='axes fraction')
    axs[1].set_yscale('log')

    axs[2].scatter(t_sample, L_3, color='blue', s=10)
    axs[2].errorbar(t_sample, L_3, yerr=L_lum_err_3, fmt='none', color='blue')
    axs[2].set_xlim(xlim)
    axs[2].axhline(L, 0, 1, color='black', linestyle='--')
    axs[2].annotate(label3, (0.82, 0.9), xycoords='axes fraction')
    axs[2].set_yscale('log')

    axs[2].set_xlabel('Time (yr)')

    plt.savefig('figures/gaia_light_curve_lum_no_lines.png', bbox_inches='tight')
    plt.savefig('figures/gaia_light_curve_lum_no_lines.pdf', bbox_inches='tight')

    axs[0].plot(t_continuous, L_1_model, 'b--')
    axs[1].plot(t_continuous, L_2_model, 'b--')
    axs[2].plot(t_continuous, L_3_model, 'b--')

    plt.savefig('figures/gaia_light_curve_lum.png', bbox_inches='tight')
    plt.savefig('figures/gaia_light_curve_lum.pdf', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(4)
    fig.set_dpi(300)

    ax.scatter(t_sample, L_1, color='blue', s=10)
    ax.errorbar(t_sample, L_1, yerr=L_lum_err_1, fmt='none', color='blue')
    ax.set_xlim(xlim)
    ax.axhline(L, 0, 1, color='black', linestyle='--')
    ax.plot(t_continuous, L_1_model, 'b--')
    ax.annotate(label1, (0.75, 0.9), xycoords='axes fraction')
    ax.set_ylabel('Luminosity ($L_{\odot}$)')
    ax.set_xlabel('Time (yr)')
    ax.set_yscale('log')

    plt.savefig('figures/gaia_light_curve_lum_1.png', bbox_inches='tight')
    plt.savefig('figures/gaia_light_curve_lum_1.pdf', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(4)
    fig.set_dpi(300)

    ax.scatter(t_sample, L_2, color='blue', s=10)
    ax.errorbar(t_sample, L_2, yerr=L_lum_err_2, fmt='none', color='blue')
    ax.set_xlim(xlim)
    ax.axhline(L, 0, 1, color='black', linestyle='--')
    ax.plot(t_continuous, L_2_model, 'b--')
    ax.annotate(label2, (0.75, 0.9), xycoords='axes fraction')
    ax.set_ylabel('Luminosity ($L_{\odot}$)')
    ax.set_xlabel('Time (yr)')
    ax.set_yscale('log')

    plt.savefig('figures/gaia_light_curve_lum_2.png', bbox_inches='tight')
    plt.savefig('figures/gaia_light_curve_lum_2.pdf', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(4)
    fig.set_dpi(300)

    ax.scatter(t_sample, L_3, color='blue', s=10)
    ax.errorbar(t_sample, L_3, yerr=L_lum_err_3, fmt='none', color='blue')
    ax.set_xlim(xlim)
    ax.axhline(L, 0, 1, color='black', linestyle='--')
    ax.plot(t_continuous, L_3_model, 'b--')
    ax.annotate(label3, (0.75, 0.9), xycoords='axes fraction')
    ax.set_ylabel('Luminosity ($L_{\odot}$)')
    ax.set_xlabel('Time (yr)')
    ax.set_yscale('log')

    plt.savefig('figures/gaia_light_curve_lum_3.png', bbox_inches='tight')
    plt.savefig('figures/gaia_light_curve_lum_3.pdf', bbox_inches='tight')
    plt.close()

    # --- Per-model stacked evolution plots (L, T, photosphere area) ---

    evo_params = [
        (light_curve_1, R_curve_1, T_curve_1, L_1, L_lum_err_1, time1[-1], label1, f'sim{i1}'),
        (light_curve_2, R_curve_2, T_curve_2, L_2, L_lum_err_2, time2[-1], label2, f'sim{i2}'),
        (light_curve_3, R_curve_3, T_curve_3, L_3, L_lum_err_3, time3[-1], label3, f'sim{i3}'),
    ]

    for lc, R_c, T_c, L_noisy, L_err, t_end, sim_label, sim_name in evo_params:
        t_evo = np.linspace(0, t_end, num=3000)
        L_evo = L + lc(t_evo)
        R_evo = R_c(t_evo)
        T_evo = T_c(t_evo)
        A_evo = 4 * np.pi * R_evo ** 2

        fig, axs = plt.subplots(3, 1, sharex=True)
        fig.set_figwidth(12)
        fig.set_figheight(9)
        fig.set_dpi(300)
        plt.subplots_adjust(hspace=0)

        axs[0].scatter(t_sample, L_noisy, color='blue', s=10)
        axs[0].errorbar(t_sample, L_noisy, yerr=L_err, fmt='none', color='blue')
        axs[0].plot(t_evo, L_evo, 'b--')
        axs[0].axhline(L, color='black', linestyle='--')
        axs[0].set_xlim([0, t_end])
        axs[0].set_yscale('log')
        axs[0].set_ylabel('Luminosity ($L_{\odot}$)')
        axs[0].annotate(sim_label, (0.55, 0.9), xycoords='axes fraction')

        axs[1].plot(t_evo, T_evo, 'b-')
        axs[1].set_xlim([0, t_end])
        axs[1].set_yscale('log')
        axs[1].set_ylabel('Temperature (K)')

        axs[2].plot(t_evo, A_evo, 'b-')
        axs[2].set_xlim([0, t_end])
        axs[2].set_yscale('log')
        axs[2].set_ylabel('Area ($R_{\\oplus}^2$)')
        axs[2].set_xlabel('Time (yr)')

        plt.savefig(f'figures/evolution_{sim_name}.png', bbox_inches='tight')
        plt.savefig(f'figures/evolution_{sim_name}.pdf', bbox_inches='tight')
        plt.close()


def monte_carlo_analysis(n_runs, n_stars, t_obs, observability_probability, chunk_size=100_000):
    """Run n_runs Monte Carlo trials simultaneously and return an array of visible impact counts.

    Stars are processed in chunks of chunk_size to vectorise over n_runs without
    allocating an (n_runs, n_stars_young) array all at once.
    """

    # 1. Stars young enough (<100 Myr) to host giant impacts, assuming ages
    #    are uniformly distributed over the 10 Gyr galactic lifetime.
    n_stars_young = int(n_stars * (100e6 / 10e9))

    # 2. Count impacts occurring within t_obs for each run.
    #    Process young stars in chunks: shape (n_runs, chunk_size) per chunk,
    #    accumulating the total across all chunks.
    total_impacts = np.zeros(n_runs, dtype=np.int64)
    stars_remaining = n_stars_young
    while stars_remaining > 0:
        size = min(chunk_size, stars_remaining)
        n_planets     = np.floor(np.random.triangular(0, 3, 8, size=(n_runs, size)))
        n_imp_life    = np.random.poisson(3, size=(n_runs, size))
        impact_rate   = (n_planets * n_imp_life) / 100e6
        impacts_chunk = np.random.poisson(impact_rate * t_obs)
        total_impacts += impacts_chunk.sum(axis=1)
        stars_remaining -= size

    # 3. Assign luminosities and test observability.
    #    total_impacts per run is O(10), so max_impacts is tiny — the mask
    #    trick to handle variable-length rows is essentially free.
    max_impacts = int(total_impacts.max())
    if max_impacts == 0:
        return np.zeros(n_runs, dtype=np.int64)

    L_all    = np.random.triangular(5e-5, 5e-3, 1e-1, size=(n_runs, max_impacts))
    prob_all = observability_probability(L_all)
    r_all    = np.random.random((n_runs, max_impacts))
    valid    = np.arange(max_impacts)[None, :] < total_impacts[:, None]
    n_visible = ((r_all < prob_all) & valid).sum(axis=1)

    return n_visible


_MC_CACHE_FILE = 'monte_carlo_cache.npz'


def run_monte_carlo(n, observability_probability, observability_probability_LSST, rerun=True):
    """Run (or load cached) Monte Carlo trials and save results to disk."""

    if os.path.exists(_MC_CACHE_FILE) and not rerun:
        cache = np.load(_MC_CACHE_FILE)
        if int(cache['n']) == n:
            print(f'Loading Monte Carlo results from {_MC_CACHE_FILE}...')
            return
        print(f'Cached n={int(cache["n"])} differs from requested n={n}, re-running...')

    print('Running LSST Monte Carlo...')
    lsst_5  = monte_carlo_analysis(n, int(0.9e9), 5,  observability_probability_LSST)
    lsst_10 = monte_carlo_analysis(n, int(0.9e9), 10, observability_probability_LSST)
    print('Running Gaia Monte Carlo...')
    gaia_5  = monte_carlo_analysis(n, int(1.8e9), 5,  observability_probability)
    gaia_10 = monte_carlo_analysis(n, int(1.8e9), 10, observability_probability)

    np.savez(_MC_CACHE_FILE, n=n, lsst_5=lsst_5, lsst_10=lsst_10, gaia_5=gaia_5, gaia_10=gaia_10)
    print(f'Monte Carlo results saved to {_MC_CACHE_FILE}')


def plot_monte_carlo():
    """Plot Monte Carlo results saved by run_monte_carlo."""

    cache = np.load(_MC_CACHE_FILE)
    lsst_5, lsst_10 = cache['lsst_5'], cache['lsst_10']
    gaia_5, gaia_10 = cache['gaia_5'], cache['gaia_10']

    fig, (ax_lsst, ax_gaia) = plt.subplots(1, 2, sharey=True, dpi=300)
    fig.subplots_adjust(wspace=0)
    fig.set_figwidth(12)
    fig.set_figheight(4)

    ax_lsst.hist(lsst_5,  bins=np.arange(0, 20), align='left', density=True, histtype='step',
                 label='5 year observation period')
    ax_lsst.hist(lsst_10, bins=np.arange(0, 20), align='left', density=True, histtype='step',
                 label='10 year observation period')
    ax_lsst.set_xlabel('Number of observable impacts')
    ax_lsst.set_ylabel('Fraction of runs')
    ax_lsst.set_title('LSST')
    ax_lsst.legend()

    ax_gaia.hist(gaia_5,  bins=np.arange(0, 20), align='left', density=True, histtype='step',
                 label='5 year observation period')
    ax_gaia.hist(gaia_10, bins=np.arange(0, 20), align='left', density=True, histtype='step',
                 label='10 year observation period')
    ax_gaia.set_xlabel('Number of observable impacts')
    ax_gaia.set_title(r'$\it{Gaia}$')
    ax_gaia.legend()

    plt.savefig('figures/monte_carlo.png', bbox_inches='tight')
    plt.savefig('figures/monte_carlo.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    # gaia_mean_flux_error_plot()
    stars, obs_prob, obs_prob_LSST = gaia_analysis_v2()
    # simulated_light_curve(stars, (16, 12))
    run_monte_carlo(300, obs_prob, obs_prob_LSST, rerun=False)
    plot_monte_carlo()


