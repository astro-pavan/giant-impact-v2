# extracts data from a snapshot and analyses it, producing a 1D photosphere model
# all values are in SI units unless otherwise specified

import jax.numpy as jnp
import matplotlib.pyplot as plt

from snapshot_analysis import snapshot

# constants and function definitions
sigma = 5.670374419e-8 # stefan-boltzmann constant
L_sun = 3.828e26 # solar luminosity in watts
R_earth = 6371000 # radius of Earth in meters
M_earth = 5.972e24 # mass of Earth in kg
day = 3600 * 24
yr = 365.25 * day
silicate_latent_heat_v = 3e7
photosphere_depth = 2/3
pi = jnp.pi
cos = lambda theta: jnp.cos(theta)
sin = lambda theta: jnp.sin(theta)

class photosphere:

    def __init__(self, filename):

        self.filename = filename
        self.snapshot = snapshot(filename)