# analyses the observability of a post impact body with gaia data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    data = []

    print('Gaia data loading...')
    for file in gaia_filenames:
        data.append(pd.read_csv(file, sep=',', skiprows=1000))

    gaia_data = pd.concat(data)
    n_stars = len(gaia_data)
    print(f'{n_stars} stars loaded')

    gaia_data['luminosity'] = luminosity(gaia_data['phot_g_mean_mag'], np.abs(gaia_data['parallax'] * 1e-3))
    gaia_data['mag_error'] = gaia_mag_error(gaia_data['phot_g_mean_mag'])

    nan_mask = ~np.isnan(gaia_data['luminosity']) & ~np.isnan(gaia_data['mag_error'])
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

    y = np.log10(lum)
    x = np.log10(min_visible_impact_luminosity)

    hist_min_vis_lum, bins = np.histogram(x, density=True, bins=bin_num)
    bin_widths = np.diff(bins)
    print(np.sum(hist_min_vis_lum * bin_widths))

    ax.hist2d(x, y, bins=bin_num, cmap='Blues')

    hist_x, bins_x, patches_x = ax_histx.hist(x, bins=bin_num, density=True)
    hist_y, bins_y, patches_y = ax_histy.hist(y, bins=bin_num, orientation='horizontal', density=True)
    #print(bins_x)
    bin_width_x, bin_widths_y = np.diff(bins_x), np.diff(bins_y)

    typical_visible_mask = bins_x[1:] < np.log10(impact_luminosity)
    max_visible_mask = bins_x[1:] < np.log10(max_impact_luminosity)

    typical_visible_frac = np.sum((hist_x * bin_width_x)[typical_visible_mask])
    max_visible_frac = np.sum((hist_x * bin_width_x)[max_visible_mask])
    print(typical_visible_frac)
    print(max_visible_frac)

    ax.set_ylim([-4.5, 2.5])
    ax.set_xlim([-6.5, 0.5])

    ax.axvline(np.log10(max_impact_luminosity), 0, 1, color='red', linestyle='--', label='Maximum Impact Luminosity')
    ax_histx.axvline(np.log10(max_impact_luminosity), 0, 1, color='red', linestyle='--')

    ax.axvline(np.log10(impact_luminosity), 0, 1, color='black', linestyle='--', label='Typical Impact Luminosity')
    ax_histx.axvline(np.log10(impact_luminosity), 0, 1, color='black', linestyle='--')

    ax.set_ylabel('$\log_{10}$[Host Star Luminosity ($L_{\odot}$)]')
    ax.set_xlabel('$\log_{10}$[Minimum Visible Impact Luminosity ($L_{\odot}$)]')
    ax.legend()

    plt.savefig('figures/gaia_luminosity_hist.pdf', bbox_inches='tight')
    plt.savefig('figures/gaia_luminosity_hist.png', bbox_inches='tight')

    #log_min_L_hist, bin_edges = np.histogram(np.log10(min_visible_impact_luminosity), bins=100, density=True)


gaia_analysis()
