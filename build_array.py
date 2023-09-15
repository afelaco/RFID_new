import matplotlib.pyplot as plt
import numpy as np
import os
import sparse
from copy import deepcopy
from lib_class import Antenna
from lib_utility import cart2sph
from numpy import max
from scipy.constants import c, pi
from tqdm import tqdm

# Paths.
path_antenna = 'data\\dipole_2.45_active'

path_array = 'data\\dipole_2.45_active_circular'
if not os.path.exists(path_array):
    os.makedirs(path_array)

# Data.
f = np.loadtxt(os.path.join(path_antenna, 'frequency.txt'))
λ = c/f
k = 2*pi/λ

# Phase centers.
N_x = 17
N_y = N_x

d_x = 0.5*λ
d_y = d_x

x_n = np.arange(N_x)*d_x
x_n = x_n-np.mean(x_n)

y_n = np.arange(N_y)*d_y
y_n = y_n-np.mean(y_n)

x_n, y_n = np.meshgrid(x_n, y_n, indexing='ij')
z_n = np.zeros(x_n.shape)

x_n, y_n, z_n = (arg.flatten() for arg in (x_n, y_n, z_n))
ρ_n, θ_n, ϕ_n = cart2sph(x_n, y_n, z_n)

R = np.max(x_n)
bool = ρ_n <= R

x_n, y_n, z_n = (arg[bool] for arg in [x_n, y_n, z_n])
ρ_n, θ_n, ϕ_n = cart2sph(x_n, y_n, z_n)

N = ρ_n.size

p_n = np.zeros((N, 3), dtype=float)

p_n[:, 0] = x_n
p_n[:, 1] = y_n
p_n[:, 2] = z_n

np.savetxt(os.path.join(path_array, 'phase_centers.txt'), p_n)

# Import library.
path_lib = 'lib\\translator.npz'
if os.path.isfile(path_lib):
    T = sparse.load_npz(path_lib)

# Unit element.
sample = Antenna(path_antenna)
sample.far_field.change_basis('cartesian')
sample.spatial_fourier_transform()

# Detect edges.
edge = (np.round(ρ_n/max(ρ_n), decimals=1) == 1)
edge = p_n[edge, :]

# Test order.
array = list()
for n in tqdm(range(edge.shape[0])):
    antenna = deepcopy(sample)
    antenna.phase_center = edge[n, :]
    antenna.translate(-antenna.phase_center, -1, T)
    array.append(antenna)

Λ_n = list(antenna.sft.L for antenna in array)
Λ = max(Λ_n)
M = (Λ+1)**2

# Translation.
array = list()
for n in tqdm(range(N)):
    antenna = deepcopy(sample)
    antenna.phase_center = p_n[n, :]
    antenna.translate(-antenna.phase_center, Λ, T)
    array.append(antenna)

# System matrix.
F = np.zeros((M, N), dtype=complex)
for n, coefficients in enumerate(antenna.sft.f for antenna in array):
    F[:, n] = np.pad(np.dot(coefficients.conj(), sample.p), (0, M-coefficients.shape[0]), 'constant')

np.savetxt(os.path.join(path_array, 'matrix.txt'), F)
