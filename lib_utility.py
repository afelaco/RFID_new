import numpy as np
from scipy.constants import pi


def cart2sph(x, y, z):
    """
    Performs the transformation from cartesian to spherical coordinates, as described `here
    <https://en.wikipedia.org/wiki/Spherical_coordinate_system>`_.
     """

    x, y, z = (np.array(arg) for arg in [x, y, z])

    θ = np.zeros(x.shape, dtype=float)
    ϕ = np.zeros(x.shape, dtype=float)

    ρ = np.sqrt(x**2+y**2+z**2)

    θ[ρ != 0] = np.arccos(z[ρ != 0]/ρ[ρ != 0])

    ϕ[x > 0] = np.arctan(y[x > 0]/x[x > 0])
    ϕ[(x < 0) & (y >= 0)] = np.arctan(y[(x < 0) & (y >= 0)]/x[(x < 0) & (y >= 0)])+pi
    ϕ[(x < 0) & (y < 0)] = np.arctan(y[(x < 0) & (y < 0)]/x[(x < 0) & (y < 0)])-pi
    ϕ[(x == 0) & (y > 0)] = pi/2
    ϕ[(x == 0) & (y < 0)] = -pi/2

    return ρ, θ, ϕ


def sph2cart(ρ, θ, ϕ):
    x = ρ*np.cos(ϕ)*np.sin(θ)
    y = ρ*np.sin(ϕ)*np.sin(θ)
    z = ρ*np.cos(θ)

    return x, y, z
