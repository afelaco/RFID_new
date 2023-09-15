import numpy as np
import os
from scipy.constants import pi


def lebedev(L):
    path_lib = 'lib\\lebedev'

    N = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131])
    N = N[(N-L) > 0][0]

    scheme = '_'.join(['lebedev', str(N).zfill(3)])
    ϕ, θ, w = (np.loadtxt(os.path.join(path_lib, scheme+'.txt'))[:, i] for i in (0, 1, 2))

    θ, ϕ = (arg*pi/180 for arg in (θ, ϕ))
    ϕ[ϕ < 0] = ϕ[ϕ < 0]+2*pi

    return θ, ϕ, w, scheme
