# analyses the observability of a post impact body with gaia data

# SELECT gaia_source.source_id,gaia_source.ra,gaia_source.dec,gaia_source.parallax,gaia_source.phot_g_mean_flux_over_error,gaia_source.phot_g_mean_mag,gaia_source.bp_rp,gaia_source.phot_variable_flag,gaia_source.non_single_star,gaia_source.has_xp_continuous,gaia_source.has_xp_sampled,gaia_source.has_rvs,gaia_source.has_mcmc_gspphot,gaia_source.has_mcmc_msc,gaia_source.teff_gspphot,gaia_source.logg_gspphot,gaia_source.distance_gspphot,gaia_source.azero_gspphot,gaia_source.ag_gspphot,gaia_source.ebpminrp_gspphot
# FROM gaiadr3.gaia_source
# WHERE
# CONTAINS(
# 	POINT('ICRS',gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),
# 	CIRCLE(
# 		'ICRS',
# 		COORD1(EPOCH_PROP_POS(310.357979753,45.280338807,2.3100,2.0100,1.8500,-4.9000,2000,2016.0)),
# 		COORD2(EPOCH_PROP_POS(310.357979753,45.280338807,2.3100,2.0100,1.8500,-4.9000,2000,2016.0)),
# 		3)
# )=1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

sun_g_band_app_mag = -26.895 # from Casagrande (2018)
sun_dist_pc = 1.495e+11 / 3.086e+16 # 1 AU in parsecs
sun_g_band_abs_mag = sun_g_band_app_mag - 5*np.log10(sun_dist_pc) + 5


# calculates the fractional change in luminosity from a change in magnitude
def fractional_luminosity_change(mag_change):
    return (10 ** (mag_change / 2.5)) - 1


# converts gaia G-band apparent magnitude into luminosity (in L_sun)
def luminosity(g_band_mag, parallax):

    g_band_abs_mag = g_band_mag + 5*np.log10(parallax) + 5
    L = 10 ** ((sun_g_band_abs_mag - g_band_abs_mag) / 2.5)

    return L


def abs_mag(g_band_mag, parallax):
    return g_band_mag + 5*np.log10(parallax) + 5


# estimates the error in G-band magnitude for a given apparent G-band magnitude
# based off the plot from the DR3 paper
def gaia_mag_error(g_band_app_mag):
    res = 0.25 * (g_band_app_mag - 17) - 3.0
    return 10 ** np.maximum(-3.5, res)


def plot_gaia_mag_error():
    m = np.linspace(10, 22)
    plt.plot(m, np.log10(gaia_mag_error(m)))
    plt.xlabel('Gaia G-band magnitude')
    plt.ylabel('$\log_{10}$[mag error]')
    plt.show()


def gaia_analysis():

    gaia_filenames = ['GaiaSource_000000-003111.csv', 'GaiaSource_003112-005263.csv', 'GaiaSource_006602-007952.csv']
    gaia_filenames = ['GaiaSource_000000-003111.csv']
    data = []

    print('Gaia data loading...')
    for file in gaia_filenames:
        data.append(pd.read_csv(file, sep=',', skiprows=1000))

    gaia_data = pd.concat(data)
    n_stars = len(gaia_data)
    print(f'{n_stars} stars loaded')

    gaia_data['luminosity'] = luminosity(gaia_data['phot_g_mean_mag'], np.abs(gaia_data['parallax'] * 1e-3))
    gaia_data['phot_g_abs_mag'] = abs_mag(gaia_data['phot_g_mean_mag'], np.abs(gaia_data['parallax'] * 1e-3))
    gaia_data['mag_error'] = gaia_mag_error(gaia_data['phot_g_mean_mag'])

    nan_mask = ~np.isnan(gaia_data['phot_g_abs_mag']) & ~np.isnan(gaia_data['bp_rp'])
    lum = gaia_data['luminosity'][nan_mask]
    mag_error = gaia_data['mag_error'][nan_mask]

    max_impact_luminosity = 1e-2
    impact_luminosity = 1e-3

    min_visible_impact_luminosity = lum * mag_error
    min_easily_visible_impact_luminosity = lum * mag_error * 10

    fig = plt.figure(figsize=(8, 8))

    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])

    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    bin_num = 400

    y = gaia_data['phot_g_abs_mag'][nan_mask]
    x = gaia_data['bp_rp'][nan_mask]

    hist_min_vis_lum, bins = np.histogram(x, density=True, bins=bin_num)
    bin_widths = np.diff(bins)
    print(np.sum(hist_min_vis_lum * bin_widths))

    ax.hist2d(x, y, bins=bin_num, cmap='inferno', norm=LogNorm())
    #ax.scatter(x, y, s=0.5)

    hist_x, bins_x, patches_x = ax_histx.hist(x, bins=bin_num, density=True)
    hist_y, bins_y, patches_y = ax_histy.hist(y, bins=bin_num, orientation='horizontal', density=True)
    #print(bins_x)
    bin_width_x, bin_widths_y = np.diff(bins_x), np.diff(bins_y)

    # typical_visible_mask = bins_x[1:] < np.log10(impact_luminosity)
    # max_visible_mask = bins_x[1:] < np.log10(max_impact_luminosity)
    #
    # typical_visible_frac = np.sum((hist_x * bin_width_x)[typical_visible_mask])
    # max_visible_frac = np.sum((hist_x * bin_width_x)[max_visible_mask])
    # print(typical_visible_frac)
    # print(max_visible_frac)
    #
    # ax.set_ylim([-4.5, 2.5])
    # ax.set_xlim([-6.5, 0.5])

    # ax.axvline(np.log10(max_impact_luminosity), 0, 1, color='red', linestyle='--', label='Maximum Impact Luminosity')
    # ax_histx.axvline(np.log10(max_impact_luminosity), 0, 1, color='red', linestyle='--')
    #
    # ax.axvline(np.log10(impact_luminosity), 0, 1, color='black', linestyle='--', label='Typical Impact Luminosity')
    # ax_histx.axvline(np.log10(impact_luminosity), 0, 1, color='black', linestyle='--')
    #
    # ax.set_ylabel('$\log_{10}$[Host Star Luminosity ($L_{\odot}$)]')
    # ax.set_xlabel('$\log_{10}$[Minimum Visible Impact Luminosity ($L_{\odot}$)]')
    # ax.legend()

    plt.savefig('figures/gaia_hist.pdf', bbox_inches='tight')
    plt.savefig('figures/gaia_hist.png', bbox_inches='tight')

    #log_min_L_hist, bin_edges = np.histogram(np.log10(min_visible_impact_luminosity), bins=100, density=True)


def gaia_HR():
    gaia_data = pd.read_csv('gaia_data.csv')

    initial_n = len(gaia_data)

    gaia_data = gaia_data[gaia_data['bp_rp'].notna()]
    gaia_data = gaia_data[gaia_data['phot_g_mean_mag'].notna()]
    gaia_data = gaia_data[gaia_data['parallax'].notna()]
    gaia_data['parallax'] = np.abs(gaia_data['parallax'])

    # np.log10(np.abs(gaia_data['parallax_over_error'])).hist(bins=200)
    # plt.show()
    #
    np.log10(gaia_data['phot_g_mean_flux_over_error']).hist(bins=200)
    plt.show()
    #
    # np.log10(gaia_data['phot_bp_mean_flux_over_error']).hist(bins=200)
    # plt.show()
    #
    # np.log10(gaia_data['phot_rp_mean_flux_over_error']).hist(bins=200)
    # plt.show()

    gaia_data = gaia_data[gaia_data['parallax_over_error'] > 3]
    # gaia_data = gaia_data[gaia_data['phot_g_mean_flux_over_error'] > 50]
    # gaia_data = gaia_data[gaia_data['phot_bp_mean_flux_over_error'] > 20]
    # gaia_data = gaia_data[gaia_data['phot_rp_mean_flux_over_error'] > 20]

    filtered_n = len(gaia_data)

    print(f'{filtered_n / initial_n:.2%} kept')

    gaia_data['abs_g_mag'] = abs_mag(gaia_data['phot_g_mean_mag'], gaia_data['parallax'] * 1e-3)

    plt.hist2d(gaia_data['phot_g_mean_mag'], np.log10(gaia_data['phot_g_mean_flux_over_error']), bins=200, cmap='inferno', norm=SymLogNorm(10))
    # plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    # plt.ylim([15, 0])
    plt.show()


gaia_HR()

