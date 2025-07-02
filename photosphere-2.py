# extracts data from a snapshot and analyses it, producing a 1D photosphere model
# all values are in SI units unless otherwise specified

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp

from swiftsimio.visualisation import slice_gas
from unyt import Rearth

from snapshot_analysis import snapshot
import EOS as fst

sigma = 5.670374419e-8 # stefan-boltzmann constant
L_sun = 3.828e26 # solar luminosity in watts
R_earth = 6371000 # radius of Earth in meters
M_earth = 5.972e24 # mass of Earth in kg
G = 6.67430e-11 # gravitational constant in m^3 kg^-1 s^-2
day = 3600 * 24
yr = 365.25 * day
silicate_latent_heat_v = 3e7
photosphere_depth = 2/3
pi = np.pi
cos = lambda theta: np.cos(theta)
sin = lambda theta: np.sin(theta)

class photosphere:

    def __init__(self, filename):

        self.filename = filename
        self.snapshot = snapshot(filename)

        resolution = 1000
        sample_size = 10 * Rearth
        max_size = 50 * 6371000

        # calculate the center of the snapshot and set the limits for the slice
        center = self.snapshot.center_of_mass
        limits = [center[0] - sample_size, center[0] + sample_size, center[1] - sample_size,
                      center[1] + sample_size]
        

        n_theta = 100
        r_range = np.linspace(0, sample_size.value * 0.95, num=int(resolution / 2)) * Rearth
        theta_range = np.arange(n_theta+1) * (pi / n_theta)
        r_2d, theta_2d = np.meshgrid(r_range, theta_range)

        pixel_size = sample_size.value / (resolution / 2)

        i_x = np.int32((r_2d.value * np.cos(theta_2d) / pixel_size) + (resolution / 2))
        i_y = np.int32((r_2d.value * np.sin(theta_2d) / pixel_size) + (resolution / 2))

        indexes = i_y, i_x

        # loads density slice
        mass_slice = slice_gas(self.snapshot.data,
                               z_slice=center[2],
                               resolution=resolution,
                               project="masses",
                               region=limits,
                               parallel=True
                               )

        # function that loads the slice of each property
        def get_slice(parameter):

            mass_weighted_slice = slice_gas(
                self.snapshot.data,
                z_slice=center[2],
                resolution=resolution,
                project=f'{parameter}_mass_weighted',
                region=limits,
                parallel=True
            )

            property_slice = mass_weighted_slice / mass_slice

            return property_slice[tuple(indexes)]
        
        mass_slice.convert_to_mks()
        
        # loading slices of each property
        densities = mass_slice[tuple(indexes)]
        temperatures = get_slice('temperatures')
        pressures, entropies = get_slice('pressures'), get_slice('entropy')

        # convert data to MKS
        temperatures.convert_to_mks()
        pressures.convert_to_mks()
        entropies.convert_to_mks()
        r_range.convert_to_mks()

        self.rho = np.array(np.mean(densities, axis=0))
        self.T = np.array(np.mean(temperatures, axis=0))
        self.P = np.array(np.mean(pressures, axis=0))
        self.s = np.array(np.mean(entropies, axis=0))
        self.r = np.array(r_range)

        r_extension = np.linspace(self.r[-1], max_size, num=resolution // 2)

        self.r = np.concatenate((self.r, r_extension))
        self.rho = np.concatenate((self.rho, np.zeros_like(r_extension)))
        self.T = np.concatenate((self.T, np.zeros_like(r_extension)))
        self.P = np.concatenate((self.P, np.zeros_like(r_extension)))
        self.s = np.concatenate((self.s, np.zeros_like(r_extension)))

        self.extrapolate_r = self.snapshot.HD_limit_R.value * 6371000  # density floor limit in meters
        self.extrapolation_index = np.argmax(self.r > self.extrapolate_r)
        s_extrapolation_value = self.s[self.extrapolation_index - 1]

        self.s = np.where(self.r < self.extrapolate_r, self.s, s_extrapolation_value)
        self.s_interpolation = CubicSpline(self.r, self.s)

        plt.plot(self.r, self.s)
        plt.savefig("entropy_profile.png")
        plt.close()

    def solve_dPdr(self):
        
        def dPdr(r, P):
            
            s = self.s_interpolation(r)
            omega = self.snapshot.best_fit_rotation_curve_mks(r)
            M = self.snapshot.total_mass 

            rho = fst.rho_EOS(s, P)
            gravity = - G * M / (r ** 2)
            centrifugal = r * omega ** 2

            return - rho * (gravity + centrifugal)
        
        solution = solve_ivp(
            dPdr,
            (self.extrapolate_r, self.r[-1]),
            [self.P[self.extrapolation_index - 1]],
            t_eval=self.r[self.extrapolation_index:]
            )
        
        self.P[self.extrapolation_index:] = solution.y[0]
        





    def calculate_EOS(self):
        pass

    def remove_droplets(self):
        pass

    def calculate_luminosity(self):
        pass

    def cool_step(self, dt):
        pass

    def cool(self, t):
        pass


if __name__ == "__main__":

    p1 = photosphere("snapshot_0240.hdf5")