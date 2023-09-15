import numpy as np
import scipy as sp


def interpolate(θ, ϕ, v, θ_i, ϕ_i):
    """ Interpolates v from (θ, ϕ) to (θ_i, ϕ_i). """

    grid = (np.unique(θ), np.unique(ϕ))
    v_i = sp.interpolate.RegularGridInterpolator(grid, v)
    grid_i = np.moveaxis(np.array([θ_i, ϕ_i]), 0, -1)

    return v_i(grid_i)
