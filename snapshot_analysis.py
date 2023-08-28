# reads snapshots of SWIFT simulations for plotting and analysis
import swiftsimio as sw
from matplotlib.colors import LogNorm, SymLogNorm
from swiftsimio.visualisation.slice import slice_gas
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
import woma
from unyt import Rearth, Pa, K, kg, J, m, s, g, cm
import unyt
unyt.define_unit("M_earth", 5.9722e24 * unyt.kg)
M_earth = unyt.M_earth

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


data_labels = {
    "z": "z ($R_{\oplus}$)",
    "R": "Cylindrical Radius ($R_{\oplus}$)",
    "rho": "Density ($g/cm^{3}$)",
    "T": "Temperature (K)",
    "P": "Pressure (Pa)",
    "s": "Specific Entropy (J/K/kg)",
    "omega": "Angular Velocity (rad/s)",
    "alpha": "Absorption ($m^{-1}$)",
    "phase": "Phase",
    "v_head": "Headwind (m/s)",
    "v_r": "Radial Velocity (m/s)",
    "c_s": "Sound Speed (m/s)",
    "u": "Specific Internal Energy (J/kg)",
    "rho_u": "Internal Energy Density ($J/m^{3}$)",
    "matid": "Material ID",
    "t_sound": "Sound crossing time (s)",
    "t_cool": "Cooling time (s)",
    "E": "Energy (J)",
    "L": "Luminosity (W)",
    "m": "Mass (kg)",
    "dL": "Luminosity (W)"
}

colormaps = {
    "rho": "magma",
    "T": "coolwarm",
    "P": "plasma",
    "S": "viridis",
    "omega": "inferno",
    "alpha": "magma",
    "phase": "magma",
    "u": "inferno",
    "matid": "viridis",
    "v_r": "seismic"
}


# class that stores and analyses particle data in a SWIFT snapshot
class snapshot:

    def __init__(self, filename):

        # loads particle data
        self.data = sw.load(filename)
        print(f'Loaded {len(self.data.gas.densities)} particles')

        self.box_size = self.data.gas.metadata.boxsize
        self.center_of_mass = self.get_center_of_mass()

        self.data.gas.masses.convert_to_mks()
        self.total_mass = np.sum(self.data.gas.masses)
        self.total_mass.convert_to_units(M_earth)
        print(f'Total mass of particles {self.total_mass:.4e}')
        self.total_mass.convert_to_mks()

        self.total_angular_momentum = 0
        self.total_specific_angular_momentum = 0

        # calculates the coordinates of the particles relative to the CoM
        pos = np.array(self.data.gas.coordinates - self.center_of_mass)
        self.R_xy = np.hypot(pos[:, 0], pos[:, 1]) * Rearth
        self.z = pos[:, 2] * Rearth
        self.r = np.hypot(np.hypot(pos[:, 0], pos[:, 1]), pos[:, 2]) * Rearth

        self.calculate_EOS()
        self.calculate_velocities()

        self.HD_limit_R, self.HD_limit_z = self.particle_density_analysis()
        self.HD_limit_R.convert_to_mks()
        self.HD_limit_z.convert_to_mks()
        self.best_fit_rotation_curve, self.best_fit_rotation_curve_mks, self.CoRoL = self.rotational_analysis()

    # calculates the EOS for all particles
    def calculate_EOS(self):
        print('Applying EOS to particles...')

        gas = self.data.gas
        woma.load_eos_tables()
        gas.internal_energies.convert_to_mks()
        gas.densities.convert_to_mks()

        u, rho, mat_id = np.array(gas.internal_energies), np.array(gas.densities), np.array(gas.material_ids)

        try:
            T = woma.A1_T_u_rho(u, rho, mat_id)
            P = woma.A1_P_u_rho(u, rho, mat_id)
            S = woma.A1_s_u_rho(u, rho, mat_id)
        except ValueError:
            gas.material_ids_mass_weighted = gas.material_ids * gas.masses
            return

        gas.temperatures = sw.objects.cosmo_array(T * K)
        gas.pressures = sw.objects.cosmo_array(P * Pa)
        gas.entropy = sw.objects.cosmo_array(S * ((J / K) / kg))

        gas.temperatures.cosmo_factor = gas.internal_energies.cosmo_factor
        gas.pressures.cosmo_factor = gas.internal_energies.cosmo_factor
        gas.entropy.cosmo_factor = gas.internal_energies.cosmo_factor

        gas.temperatures_mass_weighted = gas.temperatures * gas.masses
        gas.pressures_mass_weighted = gas.pressures * gas.masses
        gas.entropy_mass_weighted = gas.entropy * gas.masses
        gas.internal_energies_mass_weighted = gas.internal_energies * gas.masses

        gas.material_ids_mass_weighted = gas.material_ids * gas.masses

        print('EOS calculated')

    # calculates the centre of mass in the snapshot
    def get_center_of_mass(self):

        pos, densities = np.array(self.data.gas.coordinates), np.array(self.data.gas.densities)
        mass_sum, mass_pos_sum = 0, 0

        for i in range(pos.shape[0]):
            mass_pos_sum += pos[i] * densities[i]
            mass_sum += densities[i]
        center_of_mass = np.array(mass_pos_sum / mass_sum) * Rearth

        print(f'Center of mass found at {center_of_mass}')
        return center_of_mass

    # uses unyt
    def mass_within_r(self, r):

        if type(r) is np.ndarray:
            mass_total = np.zeros_like(r)
            for i in range(len(r)):
                mass_total[i] = np.sum(self.data.gas.masses[self.r < r[i]])
            result = mass_total * kg
        else:
            self.data.gas.masses.convert_to_mks()
            result = np.sum(self.data.gas.masses[self.r < r])

        result.convert_to_units(M_earth)
        return result

    # calculates the EOS for all particles    def calculate_EOS(self):
        print('Applying EOS to particles...')

        gas = self.data.gas
        woma.load_eos_tables()
        gas.internal_energies.convert_to_mks()
        gas.densities.convert_to_mks()

        u, rho, mat_id = np.array(gas.internal_energies), np.array(gas.densities), np.array(gas.material_ids)

        try:
            T = woma.A1_T_u_rho(u, rho, mat_id)
            P = woma.A1_P_u_rho(u, rho, mat_id)
            S = woma.A1_s_u_rho(u, rho, mat_id)
        except ValueError:
            gas.material_ids_mass_weighted = gas.material_ids * gas.masses
            return

        gas.temperatures = sw.objects.cosmo_array(T * K)
        gas.pressures = sw.objects.cosmo_array(P * Pa)
        gas.entropy = sw.objects.cosmo_array(S * ((J / K) / kg))

        gas.temperatures.cosmo_factor = gas.internal_energies.cosmo_factor
        gas.pressures.cosmo_factor = gas.internal_energies.cosmo_factor
        gas.entropy.cosmo_factor = gas.internal_energies.cosmo_factor

        gas.temperatures_mass_weighted = gas.temperatures * gas.masses
        gas.pressures_mass_weighted = gas.pressures * gas.masses
        gas.entropy_mass_weighted = gas.entropy * gas.masses
        gas.internal_energies_mass_weighted = gas.internal_energies * gas.masses

        gas.material_ids_mass_weighted = gas.material_ids * gas.masses

        print('EOS calculated')

    # calculates the vertical, radial and angular velocities of the particles as well as the angular momentum
    def calculate_velocities(self):
        gas = self.data.gas

        # changes the units to MKS for calculations as some vector operations remove the unit
        gas.coordinates.convert_to_mks()
        gas.velocities.convert_to_mks()
        self.center_of_mass.convert_to_mks()

        masses = gas.masses
        r, v = np.array(gas.coordinates - self.center_of_mass), np.array(gas.velocities)

        # h = np.cross(r, v)[:, 2] * ((m ** 2)/s)
        h = (r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]) * ((m ** 2)/s)
        omega = h / (self.R_xy ** 2)

        v_r = np.sum(v * r, axis=1) / np.sqrt(np.sum(r * r, axis=1)) * (m/s)
        v_z = v[:, 2] * np.sign(r[:, 2]) * (m/s)

        # puts the calculated velocities into SWIFT arrays

        gas.angular_velocity = sw.objects.cosmo_array(omega)
        gas.angular_velocity.cosmo_factor = gas.internal_energies.cosmo_factor
        gas.angular_velocity_mass_weighted = gas.angular_velocity * gas.masses

        gas.specific_angular_momentum = sw.objects.cosmo_array(h)
        gas.specific_angular_momentum.cosmo_factor = gas.internal_energies.cosmo_factor
        gas.specific_angular_momentum_mass_weighted = gas.specific_angular_momentum * gas.masses

        self.total_angular_momentum = np.sum(h * masses)
        self.total_angular_momentum.convert_to_mks()
        print(f'Total angular momentum of particles {self.total_angular_momentum:.4e}')

        self.total_specific_angular_momentum = np.sum(h) * ((m ** 2)/s)
        self.total_specific_angular_momentum.convert_to_mks()
        print(f'Total specific angular momentum of particles {self.total_specific_angular_momentum:.4e}')

        gas.radial_velocity = sw.objects.cosmo_array(v_r)
        gas.radial_velocity.cosmo_factor = gas.internal_energies.cosmo_factor
        gas.radial_velocity_mass_weighted = gas.radial_velocity * gas.masses

        gas.vertical_velocity = sw.objects.cosmo_array(v_z)
        gas.vertical_velocity.cosmo_factor = gas.internal_energies.cosmo_factor
        gas.vertical_velocity_mass_weighted = gas.vertical_velocity * gas.masses

        self.center_of_mass.convert_to_units(Rearth)

    # find the regions in the snapshot where the particle density is sufficient to analyse
    def particle_density_analysis(self):

        # sets up the particle distribution histogram as a function of radius
        R_bins = np.logspace(-2, 2, num=50)
        R_hist = np.histogram(self.R_xy, R_bins)

        # R_hist_x is the outer radius of the bin, R_hist_y is the particle area density in the bin
        R_hist_x, R_hist_y = np.zeros_like(R_hist[0], dtype=float), np.array(R_hist[0], dtype=float)

        # calculates the particle area density for each bin
        for i in range(len(R_hist[0])):
            R_in, R_out = R_hist[1][i], R_hist[1][i + 1]
            area = np.pi * (R_out ** 2 - R_in ** 2)
            R_hist_y[i], R_hist_x[i] = R_hist_y[i] / area, R_out

        # finds the radius at which the particle density drops below a certain density (in particles per Rearth ** 2)
        critical_density = 3
        R_HD_region_mask = R_hist_y > critical_density
        R_HD_limit = R_bins[np.argmin(R_HD_region_mask) - 1]

        # sets up the histogram as a function of height
        n_z = 100
        z_bins = np.array((self.box_size[2] / n_z) * np.arange(-n_z, n_z + 1))
        z_hist = np.histogram(self.z.value, z_bins)
        z_area = np.array(self.box_size[0] * (self.box_size[2] / n_z))

        # populates the histogram array
        z_hist_x, z_hist_y = np.array(z_hist[1][:-1], dtype=float), np.array(z_hist[0] / z_area, dtype=float)

        # finds the heights at which the particle density drops below a certain density (in particles per Rearth ** 2)
        z_HD_mask_min = (z_hist_y > critical_density) & (z_hist_x < 0)
        z_HD_mask_max = (z_hist_y < critical_density) & (z_hist_x > 0)
        z_HD_min, z_HD_max = z_bins[np.argmax(z_HD_mask_min)], z_bins[np.argmax(z_HD_mask_max)]

        # gets the average of the heights
        z_HD_limit = (np.abs(z_HD_min) + np.abs(z_HD_max)) / 2

        return R_HD_limit * Rearth, z_HD_limit * Rearth

    # analyses the rotation of the particles to produce a best fit rotation curve
    def rotational_analysis(self, plot_output=False, save=False):

        # gets the particles in a valid region and takes the log of the cylindrical radius and angular velocity
        midplane_mask = (np.abs(self.z) < 0.5 * Rearth) & (self.R_xy < self.HD_limit_R)
        log_R, log_omega = np.log10(self.R_xy[midplane_mask]), np.log10(self.data.gas.angular_velocity[midplane_mask])

        # removes invalid values (NaN and inf)
        nan_inf_mask = np.isnan(log_R) | np.isnan(log_omega) | np.isinf(log_R) | np.isinf(log_omega)
        log_R, log_omega = log_R[~nan_inf_mask], log_omega[~nan_inf_mask]

        # the model used to fit the particle rotation
        # has a constant co-rotating inner section and a power law outer section
        def two_lines(x, a, b, c):
            constant = a
            linear = a - c * (x - b)
            return np.minimum(constant, linear)

        # fits the particle rotation to the model
        try:
            fit = curve_fit(two_lines, log_R, log_omega, p0=(-3.3, 0, 1.8))
            a0, b0, c0 = fit[0][0], fit[0][1], fit[0][2]
        except TypeError:
            print('ERROR: UNABLE TO MODEL OMEGA')
            a0, b0, c0 = 0, 0, 0

        # rotation curve function (uses unyt)
        def best_fit(R):
            R.convert_to_units(Rearth)
            return (10 ** (two_lines(np.log10(R.value), a0, b0, c0))) * (s ** -1)

        def best_fit_mks(R):
            return (10 ** (two_lines(np.log10(R * 6371000), a0, b0, c0)))

        CoRoL = b0 * Rearth

        omega_keplerian = lambda R: np.sqrt((6.674e-11 * (self.total_mass.value / 5.9722e24)) / ((R * 6371000) ** 3))

        x2 = np.logspace(b0, 2)
        x1 = np.logspace(-2, b0)

        if plot_output:

            rand = np.random.random(len(self.R_xy))
            plot_mask = (((rand < 0.02) & (self.R_xy < 1)) | ((rand < 0.3) & (self.R_xy > 1))) & \
                        (np.abs(self.z) < 0.5 * Rearth)

            plt.scatter(self.R_xy[plot_mask], self.data.gas.angular_velocity[plot_mask], s=0.2, c='blue', marker='o')
            plt.plot(x2, best_fit(x2), linestyle='--', color='red', label='Best fit rotation curve')
            plt.plot(x2, omega_keplerian(x2), linestyle='--', color='black', label='Keplerian rotation curve')
            plt.plot(x1, np.full_like(x1, 10 ** a0), 'r--')
            plt.xlabel('Cyl. Radius ($R_{\oplus}$)')
            plt.ylabel('Angular velocity (rad/s)')
            plt.axvspan(10, 100, alpha=0.5, color='grey')
            plt.xlim([1e-1, 1e2])
            plt.ylim([1e-8, 1e-2])
            plt.legend()
            plt.xscale('log')
            plt.yscale('log')

            if save:
                plt.savefig('plots/plot.png', bbox_inches='tight')
            else:
                plt.show()

        return best_fit, best_fit_mks, CoRoL


# class that stores a 2D slice of the SWIFT snapshot used for plotting and analysis
class gas_slice:

    def __init__(self, snapshot, resolution=1024, size=1, center=(0, 0, 0), rotate_vector=(0, 0, 1)):

        size = size * 2
        self.snapshot = snapshot
        self.resolution = resolution
        self.size = size * Rearth
        self.x_offset = center[1]
        self.y_offset = center[0]

        rotate_z = rotation_matrix_from_vector(rotate_vector, axis='z')
        rotate_x = rotation_matrix_from_vector(rotate_vector, axis='x')
        self.matrix = np.matmul(rotate_x, rotate_z)

        self.center = center * Rearth + self.snapshot.center_of_mass  # center of snapshot relative to the box coords
        self.limits = [self.center[0] - self.size / 2,
                       self.center[0] + self.size / 2,
                       self.center[1] - self.size / 2,
                       self.center[1] + self.size / 2]  # edges of snapshot relative to box coords

        self.pixels_per_Rearth = int(self.resolution / self.size)

        self.center_of_mass_index = 0

        self.matrix = rotation_matrix_from_vector(rotate_vector, axis='z')

        self.data = {}

        z_slice = center[2] * Rearth
        z_slice = 0

        self.data['rho'] = slice_gas(
            self.snapshot.data,
            z_slice=z_slice,
            resolution=self.resolution,
            project="masses",
            region=self.limits,
            rotation_matrix=self.matrix,
            rotation_center=self.snapshot.center_of_mass,
            parallel=True
        )

        self.data['rho'].convert_to_units(kg / m ** 3)

        def get_slice(parameter):
            mass_weighted_slice = slice_gas(
                self.snapshot.data,
                z_slice=z_slice,
                resolution=self.resolution,
                project=f'{parameter}_mass_weighted',
                region=self.limits,
                rotation_matrix=self.matrix,
                rotation_center=self.snapshot.center_of_mass,
                parallel=True
            )

            return mass_weighted_slice / self.data['rho']

        self.data['matid'] = get_slice("material_ids") / 400

        try:
            self.data['T'] = get_slice("temperatures")
            self.data['P'] = get_slice("pressures")
            self.data['S'] = get_slice("entropy")
            self.data['omega'] = get_slice("angular_velocity")
            self.data['v_r'] = get_slice("radial_velocity")
            self.data['u'] = get_slice("internal_energies")
        except AttributeError:
            pass

        self.data['rho'].convert_to_units(g / cm ** 3)

    def ticks(self):

        factor = 10 ** (int(np.log10(self.size)) - 1)

        a = int(self.size / 2)
        x_tick_pos, x_tick_label = [], []
        y_tick_pos, y_tick_label = [], []

        for i in range(-a, a + 1):
            x_tick_pos.append(self.resolution / 2 - i * self.pixels_per_Rearth / factor)
            if (i - self.x_offset) % 2.5 == 0 or self.size < 6 * Rearth / factor:
                x_tick_label.append(-(i - self.x_offset) / factor)
            else:
                x_tick_label.append('')

        for i in range(-a, a + 1):
            y_tick_pos.append(self.resolution / 2 - i * self.pixels_per_Rearth / factor)
            if (i - self.y_offset) % 2.5 == 0 or self.size < 6 * Rearth / factor:
                y_tick_label.append((i - self.y_offset) / factor)
            else:
                y_tick_label.append('')

        return x_tick_pos, x_tick_label, y_tick_pos, y_tick_label

    # plots a heatmap for a certain parameter
    def plot(self, parameter, show=True, save=None, log=True, threshold=None):

        fig, ax = plt.subplots()
        x_tick_pos, x_tick_label, y_tick_pos, y_tick_label = self.ticks()

        if threshold is not None:
            img = ax.imshow(self.data[parameter].value, cmap=colormaps[parameter], norm=SymLogNorm(linthresh=threshold))
        elif log:
            img = ax.imshow(self.data[parameter].value, cmap=colormaps[parameter], norm=LogNorm())
        else:
            img = ax.imshow(self.data[parameter].value, cmap=colormaps[parameter])

        fig.colorbar(img, label=data_labels[parameter])

        ax.set_xticks(x_tick_pos)
        ax.set_xticklabels(x_tick_label)
        ax.set_xlabel('X ($R_{\\oplus}$)')
        ax.set_yticks(y_tick_pos)
        ax.set_yticklabels(y_tick_label)
        ax.set_ylabel('Y ($R_{\\oplus}$)')
        ax.tick_params(axis="x", direction="in", color="white")
        ax.tick_params(axis="y", direction="in", color="white")

        if save is not None:
            plt.savefig(save, bbox_inches='tight')
        if show:
            plt.show()

        plt.close()

    # produces a plot of rho, T, P, S
    def full_plot(self, show=True, save=None):

        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        x_tick_pos, x_tick_label, y_tick_pos, y_tick_label = self.ticks()

        def plot_single(ax, parameter):

            if parameter == 'rho':
                img = ax.imshow(self.data[parameter].value, cmap=colormaps[parameter], norm=SymLogNorm(linthresh=1e-4))
            elif parameter == 'T':
                img = ax.imshow(self.data[parameter].value, cmap=colormaps[parameter], norm=LogNorm(vmax=10000))
            else:
                img = ax.imshow(self.data[parameter].value, cmap=colormaps[parameter], norm=LogNorm())

            fig.colorbar(img, label=data_labels[parameter])
            ax.set_xticks(x_tick_pos)
            ax.set_xticklabels(x_tick_label)
            ax.set_xlabel('X ($R_{\\oplus}$)')
            ax.set_yticks(y_tick_pos)
            ax.set_yticklabels(y_tick_label)
            ax.set_ylabel('Y ($R_{\\oplus}$)')
            ax.tick_params(axis="x", direction="in", color="white")
            ax.tick_params(axis="y", direction="in", color="white")

        plot_single(ax[0, 0], 'rho')
        plot_single(ax[1, 0], 'T')
        plot_single(ax[0, 1], 'P')
        plot_single(ax[1, 1], 's')

        if save is not None:
            plt.savefig(save)
        if show:
            plt.show()

        plt.close()


def test():

    snapshot1 = snapshot('snapshots/basic_twin/snapshot_0411.hdf5')
    print(snapshot1.total_mass)
    print(snapshot1.mass_within_r(2 * Rearth))
    m = 1*unyt.M_earth
    m.convert_to_mks()
    slice1 = gas_slice(snapshot1, size=5)
    slice1.plot('v_r', log=False)


def AM_plot():

    snapshot1 = snapshot('snapshots/basic_twin/snapshot_0411.hdf5')
    print(snapshot1.total_angular_momentum)

    snapshots = ['snapshots/low_mass_twin/snapshot_0274.hdf5',
                 'snapshots/basic_twin/snapshot_0411.hdf5',
                 'snapshots/high_mass_twin/snapshot_0360.hdf5',
                 'snapshots/basic_spin/snapshot_0247.hdf5',
                 'snapshots/advanced_spin/snapshot_0316.hdf5']
    L = np.array([4.7e-4, 2e-3, 3e-3, 4.3e-4, 4.5e-4])
    Q = [6759163.58216159, 9667334.53721656, 17032024.95369644, 9882334.67993462, 9044909.88212921]
    AM = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    SAM = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    for i in range(len(snapshots)):
        snapshot1 = snapshot(snapshots[i])
        snapshot1.total_mass.convert_to_mks()
        AM[i] = snapshot1.total_angular_momentum
        SAM[i] = snapshot1.total_specific_angular_momentum

    colours = ['red', 'orange', 'gold', 'green', 'blue']

    plt.scatter(SAM, Q, c=colours, marker='x')
    plt.xlabel('Specific angular momentum ($m^{2}s^{-1}$)')
    plt.ylabel('Modified specific impact energy ($J/kg$)')
    plt.show()

    plt.scatter(SAM, L, c=colours, marker='x')
    plt.xlabel('Specific angular momentum ($m^{2}s^{-1}$)')
    plt.ylabel('Luminosity ($L_{\odot}$)')
    plt.show()

    plt.scatter(Q, L, c=colours, marker='x')
    plt.xlabel('Modified specific impact energy ($J/kg$)')
    plt.ylabel('Luminosity ($L_{\odot}$)')
    plt.show()
