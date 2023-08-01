# reads snapshots of SWIFT simulations for plotting and analysis
import swiftsimio as sw
from matplotlib.colors import LogNorm, SymLogNorm
from swiftsimio.visualisation.slice import slice_gas
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
import woma
from unyt import Rearth, Pa, K, kg, J, m, s, g, cm
import unyt
unyt.define_unit("M_earth", 5.9722e24 * unyt.kg)
M_earth = unyt.M_earth

import numpy as np
import matplotlib.pyplot as plt


data_labels = {
    "z": "z ($R_{\oplus}$)",
    "R": "Cylindrical Radius ($R_{\oplus}$)",
    "rho": "Density ($g/cm^{3}$)",
    "T": "Temperature (K)",
    "P": "Pressure (Pa)",
    "S": "Specific Entropy (J/K/kg)",
    "omega": "Angular Velocity (rad/s)",
    "alpha": "Absorption ($m^{-1}$)",
    "phase": "Phase",
    "v_head": "Headwind (m/s)",
    "v_r": "Radial Velocity (m/s)",
    "c_s": "Sound Speed (m/s)",
    "u": "Specific Internal Energy (J/kg)",
    "rho_u": "Internal Energy Density ($J/m^{3}$)",
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

        # calculates the coordinates of the particles relative to the CoM
        pos = np.array(self.data.gas.coordinates - self.center_of_mass)
        self.R_xy = np.hypot(pos[:, 0], pos[:, 1]) * Rearth
        self.z = pos[:, 2] * Rearth
        self.r = np.hypot(np.hypot(pos[:, 0], pos[:, 1]), pos[:, 2]) * Rearth

        self.calculate_EOS()
        self.calculate_velocities()

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

    # calculates the EOS for all particles
    def calculate_EOS(self):
        gas = self.data.gas

        print('Applying EOS to particles...')

        woma.load_eos_tables()
        gas.internal_energies.convert_to_mks()
        gas.densities.convert_to_mks()

        u, rho, mat_id = np.array(gas.internal_energies), np.array(gas.densities), np.array(gas.material_ids)

        T = woma.eos.eos.A1_T_u_rho(u, rho, mat_id)
        P = woma.eos.eos.A1_P_u_rho(u, rho, mat_id)
        S = woma.eos.eos.A1_s_u_rho(u, rho, mat_id)

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

        r, v = np.array(gas.coordinates - self.center_of_mass), np.array(gas.velocities)

        h = np.cross(r, v)[:, 2] * ((m ** 2)/s)
        omega = h / (self.R_xy ** 2)

        v_r = np.sum(v * r, axis=1) / np.sqrt(np.sum(r * r, axis=1)) * (m/s)
        v_z = v[:, 2] * np.sign(r[:, 2]) * (m/s)

        # puts the calculated velocities into SWIFT arrays

        gas.angular_velocity = sw.objects.cosmo_array(omega)
        gas.angular_velocity.cosmo_factor = gas.internal_energies.cosmo_factor
        gas.angular_velocity_mass_weighted = gas.angular_velocity * gas.masses

        gas.angular_momentum = sw.objects.cosmo_array(h)
        gas.angular_momentum.cosmo_factor = gas.internal_energies.cosmo_factor
        gas.angular_momentum_mass_weighted = gas.angular_momentum * gas.masses

        gas.radial_velocity = sw.objects.cosmo_array(v_r)
        gas.radial_velocity.cosmo_factor = gas.internal_energies.cosmo_factor
        gas.radial_velocity_mass_weighted = gas.radial_velocity * gas.masses

        gas.vertical_velocity = sw.objects.cosmo_array(v_z)
        gas.vertical_velocity.cosmo_factor = gas.internal_energies.cosmo_factor
        gas.vertical_velocity_mass_weighted = gas.vertical_velocity * gas.masses

        self.center_of_mass.convert_to_units(Rearth)


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

        self.data['T'] = get_slice("temperatures")
        self.data['P'] = get_slice("pressures")
        self.data['S'] = get_slice("entropy")
        self.data['omega'] = get_slice("angular_velocity")
        self.data['v_r'] = get_slice("radial_velocity")
        self.data['u'] = get_slice("internal_energies")
        self.data['matid'] = get_slice("material_ids") - 400

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
        plot_single(ax[1, 1], 'S')

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


test()
