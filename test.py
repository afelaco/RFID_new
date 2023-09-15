import adaptive
import matplotlib.pyplot as plt
import numpy as np
import os
import pyvista as pv
import random
import scipy as sp
import time
from lib_function import spherical_hankel_wave as shw
from lib_synthesis import spherical_hankel_waves_synthesis as shws
from lib_utility import cart2sph
from scipy.constants import c, pi
from scipy.interpolate import RegularGridInterpolator as rgi

# Options.
optimization_option = True
save_option = True

# Paths.
path_array = 'data\\dipole_1_active_circular'
tag_array = os.path.basename(path_array)

path_pattern = 'patterns\\bin_14.png'
tag_pattern = os.path.basename(os.path.splitext(path_pattern)[0])

path_images = '\\'.join(('images', tag_array, tag_pattern, time.strftime("%y%m%d_%H%M%S")))

# Data.
f = np.loadtxt(os.path.join(path_array, 'frequency.txt'))
λ = c/f
k = 2*pi/λ

# Import phase centers.
p_n = np.loadtxt(os.path.join(path_array, 'phase_centers.txt'))

x_n, y_n, z_n = (p_n[:, i] for i in (0, 1, 2))
ρ_n, θ_n, ϕ_n = cart2sph(x_n, y_n, z_n)

r = np.max(ρ_n)
N = ρ_n.size

# Import matrix.
F = np.loadtxt(os.path.join(path_array, 'matrix.txt'), dtype=complex)
M = F.shape[0]
Λ = int(np.sqrt(M)-1)

# Import target pattern.
x_min = -4*λ
x_max = 4*λ
n_λ_x = int((x_max-x_min)/λ)

y_min = -4*λ
y_max = 4*λ
n_λ_y = int((y_max-y_min)/λ)

z_min = 0
z_max = 12*λ
n_λ_z = int((z_max-z_min)/λ)

pattern = plt.imread(path_pattern)
pattern = np.linalg.norm(pattern, axis=-1)
pattern = np.round((np.max(pattern)-pattern)/(np.max(pattern)-np.min(pattern)))

x = np.linspace(x_min, x_max, pattern.shape[1])
y = -np.linspace(y_min, y_max, pattern.shape[0])
z = z_max/2

pattern_interpolator = rgi((y, x), pattern)

# Regular sampling.
s_p_λ = 4

x_o = np.linspace(x_min, x_max, s_p_λ*n_λ_x+1)
y_o = -np.linspace(y_min, y_max, s_p_λ*n_λ_y+1)
x_o, y_o = np.meshgrid(x_o, y_o)
x_o, y_o = (arg.flatten() for arg in [x_o, y_o])
z_o = z*np.ones(x_o.shape)

# Adaptive sampling.
O = x_o.size


def fun(xy):
    _x, _y = xy
    return pattern_interpolator([_y, _x])


learner = adaptive.Learner2D(fun, bounds=((np.min(x), np.max(x)), (np.min(y), np.max(y))))
runner = adaptive.runner.simple(learner, npoints_goal=O)

x_o_adp = np.array(learner.to_numpy())[:, 0]
y_o_adp = np.array(learner.to_numpy())[:, 1]
z_o_adp = z*np.ones(x_o_adp.shape)

x_o = np.concatenate((x_o, x_o_adp), axis=0)
y_o = np.concatenate((y_o, y_o_adp), axis=0)
z_o = np.concatenate((z_o, z_o_adp), axis=0)

# Filter.
# x_o_avg = list()
# y_o_avg = list()
# for idx_x in range(-16, 17):
#     for idx_y in range(-16, 17):
#         x_o_to_avg = x_o[(x_o > (idx_x-1)*λ/4+λ/8) & (x_o <= idx_x*λ/4+λ/8) & (y_o > (idx_y-1)*λ/4+λ/8) & (y_o <= idx_y*λ/4+λ/8)]
#         y_o_to_avg = y_o[(x_o > (idx_x-1)*λ/4+λ/8) & (x_o <= idx_x*λ/4+λ/8) & (y_o > (idx_y-1)*λ/4+λ/8) & (y_o <= idx_y*λ/4+λ/8)]
#         x_o_avg.append(np.mean(x_o_to_avg))
#         y_o_avg.append(np.mean(y_o_to_avg))
#
# x_o = np.array(x_o_avg).copy()
# y_o = np.array(y_o_avg).copy()
# z_o = z*np.ones(x_o.shape)

# Interpolate.
e = pattern_interpolator(np.array([y_o, x_o]).T)
e[e >= 0.5] = 1
e[e < 0.5] = 0
O = e.size
bool_in = e == 1
e = e[bool_in]

# H matrix.
H = np.zeros((O, M), dtype=complex)
ρ_o, θ_o, ϕ_o = cart2sph(x_o, y_o, z_o)
for l in range(Λ+1):
    for m in range(-l, l+1):
        H[:, l*(l+1)+m] = shw(k*ρ_o, θ_o, ϕ_o, l, m)

Z = -1j*k*H@F
ζ = Z[bool_in, ...]
Γ = Z[~bool_in, ...]
Y = np.linalg.inv(ζ.conj().T@ζ+Γ.conj().T@Γ)@ζ.conj().T
κ = np.linalg.cond(Y)

# SVD.
# U, σ, V_H = np.linalg.svd(Y)
# U_inv = np.linalg.pinv(U)
#
# Σ = np.zeros((N, O))
# for idx in range(min(N, O)):
#     Σ[idx, idx] = σ[idx]
#
# # Eigenvalues.
# plt.figure()
# plt.plot(20*np.log10(σ))
#
# # Eigenfields.
# idx = 0
#
# e_σ = np.reshape(V_H[idx, :], (33, 33))
#
# plt.figure()
# plt.imshow(np.abs(e_σ), interpolation='spline36')
#
# # Eigencurrents.
# i_σ = U_inv[idx, :]
#
# plt.figure()
# plt.scatter(x_n/λ, y_n/λ, c=np.abs(i_σ), cmap='OrRd', vmin=0)
# plt.xlabel('x [λ]')
# plt.ylabel('y [λ]')
# plt.gca().set_aspect('equal')
# cbar = plt.colorbar()
# cbar.ax.set_title('A')
#
# plt.figure()
# plt.scatter(x_n/λ, y_n/λ, c=np.angle(i_σ)/pi*180, cmap='OrRd', vmin=-180, vmax=180)
# plt.xlabel('x [λ]')
# plt.ylabel('y [λ]')
# plt.gca().set_aspect('equal')
# cbar = plt.colorbar(ticks=[-180, -90, 0, 90, 180])
# cbar.ax.set_title('deg')
# plt.set_cmap('twilight_shifted')

# Images folder.
if ~os.path.exists(path_images) and save_option:
    os.makedirs(path_images)

# Currents.
i = Y@e

# Optimization.
if optimization_option:
    x_0 = np.zeros(N+1, dtype=float)
    x_0[0] = np.mean(np.abs(i))
    x_0[1:] = np.angle(i)

    ε_opt = list()


    def callback(_x):
        _i = _x[0]*np.exp(1j*_x[1:])
        ε_opt.append(np.linalg.norm(ζ@_i-e)**2+np.linalg.norm(Γ@_i)**2)
        print('Error = %.1f' %ε_opt[-1], end='\r')


    def fun(_x):
        _i = _x[0]*np.exp(1j*_x[1:])
        _ε = np.linalg.norm(ζ@_i-e)**2+np.linalg.norm(Γ@_i)**2

        return _ε


    result = sp.optimize.minimize(fun, x_0, callback=callback, method='Powell')
    i = result.x[0]*np.exp(1j*result.x[1:])

a = np.abs(i)
ϕ = np.angle(i)
ε = np.linalg.norm(ζ@i-e)**2+np.linalg.norm(Γ@i)**2

# Failing elements.
# failure_rate = 5
# fail = np.random.randint(0, N, int(failure_rate*N/100))
# i = np.delete(i, fail, axis=0)
# x_n = np.delete(x_n, fail, axis=0)
# y_n = np.delete(y_n, fail, axis=0)
# Z = np.delete(Z, fail, axis=1)

# Magnitude uncertainty.
# σ_mag = 0.001
# a = np.random.normal(a, σ_mag)
# i = a*np.exp(1j*ϕ)

# Phase uncertainty.
# σ_phs = pi/8
# ϕ = np.random.normal(ϕ, σ_phs)
# i = a*np.exp(1j*ϕ)

# if save_option:
#     # Iterations.
#     I = 100
#
#     # Stability to failure.
#     failure_rate = np.linspace(0, 50)
#     ε_fail = np.empty((failure_rate.size, I))
#     for idx_rate, rate in enumerate(failure_rate):
#         for idx_iter in range(I):
#             fail = random.sample(range(N), int(rate/100*N))
#             i_fail = np.delete(i, fail, axis=0)
#             ζ_fail = np.delete(ζ, fail, axis=1)
#             Γ_fail = np.delete(Γ, fail, axis=1)
#             ε_fail[idx_rate, idx_iter] = np.linalg.norm(ζ_fail@i_fail-e)**2+np.linalg.norm(Γ_fail@i_fail)**2
#
#     plt.figure()
#     plt.errorbar(failure_rate, (np.mean(ε_fail, axis=1)-ε)/ε, np.ptp(ε_fail, axis=1)/2/ε, fmt='-o')
#     plt.title('$\epsilon_{o} = %.1f$' %ε)
#     plt.xlim([0, np.max(failure_rate)])
#     plt.xlabel('Failure rate [%]')
#     plt.gca().set_ylim(bottom=0)
#     plt.ylabel('$\Delta \epsilon / \epsilon_{o}$')
#     plt.savefig(os.path.join(path_images, 'error_fail.png'), bbox_inches='tight', transparent=True)
#
#     # Stability to magnitude deviation.
#     σ_mag = np.linspace(0, 0.001)
#     ε_mag = np.empty((σ_mag.size, I))
#     for idx_mag, σ in enumerate(σ_mag):
#         for idx_iter in range(I):
#             a_mag = np.random.normal(a, σ)
#             i_mag = a_mag*np.exp(1j*ϕ)
#             ε_mag[idx_mag, idx_iter] = np.linalg.norm(ζ@i_mag-e)**2+np.linalg.norm(Γ@i_mag)**2
#
#     plt.figure()
#     plt.errorbar(σ_mag*1000, (np.mean(ε_mag, axis=1)-ε)/ε, np.ptp(ε_mag, axis=1)/2/ε, fmt='-o')
#     plt.title('$\epsilon_{o} = %.1f$'%ε)
#     plt.xlim([0, np.max(σ_mag*1000)])
#     plt.xlabel('σ [mA]')
#     plt.gca().set_ylim(bottom=0)
#     plt.ylabel('$\Delta \epsilon / \epsilon_{o}$')
#     plt.savefig(os.path.join(path_images, 'error_mag.png'), bbox_inches='tight', transparent=True)
#
#     # Stability to phase deviation.
#     σ_phs = np.linspace(0, pi/4)
#     ε_phs = np.empty((σ_phs.size, I))
#     for idx_phs, σ in enumerate(σ_phs):
#         for idx_iter in range(I):
#             ϕ_phs = np.random.normal(ϕ, σ)
#             i_phs = a*np.exp(1j*ϕ_phs)
#             ε_phs[idx_phs, idx_iter] = np.linalg.norm(ζ@i_phs-e)**2+np.linalg.norm(Γ@i_phs)**2
#
#     plt.figure()
#     plt.errorbar(σ_phs/pi*180, (np.mean(ε_phs, axis=1)-ε)/ε, np.ptp(ε_phs, axis=1)/2/ε, fmt='-o')
#     plt.title('$\epsilon_{o} = %.1f$'%ε)
#     plt.xlim([0, np.max(σ_phs)/pi*180])
#     plt.xlabel('σ [deg]')
#     plt.gca().set_ylim(bottom=0)
#     plt.ylabel('$\Delta \epsilon / \epsilon_{o}$')
#     plt.savefig(os.path.join(path_images, 'error_phs.png'), bbox_inches='tight', transparent=True)
#
#     plt.figure()
#     plt.errorbar(failure_rate, (np.mean(ε_fail, axis=1)-ε)/ε, np.ptp(ε_fail, axis=1)/2/ε, fmt='-ro')
#     plt.title('$\epsilon_{o} = %.1f$'%ε)
#     plt.xlim([0, np.max(failure_rate)])
#     plt.xlabel('Failure rate [%]')
#     plt.ylim([0, 2])
#     plt.gca().set_ylim(bottom=0)
#     plt.ylabel('$\Delta \epsilon / \epsilon_{o}$')
#     plt.savefig(os.path.join(path_images, 'error_1.png'), bbox_inches='tight', transparent=True)
#
#     plt.figure()
#     plt.errorbar(σ_mag*1000, (np.mean(ε_fail, axis=1)-ε)/ε, np.ptp(ε_fail, axis=1)/2/ε, fmt='-ro')
#     plt.errorbar(σ_mag*1000, (np.mean(ε_mag, axis=1)-ε)/ε, np.ptp(ε_mag, axis=1)/2/ε, fmt='-bo')
#     plt.title('$\epsilon_{o} = %.1f$'%ε)
#     plt.xlim([0, np.max(σ_mag*1000)])
#     plt.xlabel('σ [mA]')
#     plt.ylim([0, 2])
#     plt.gca().set_ylim(bottom=0)
#     plt.ylabel('$\Delta \epsilon / \epsilon_{o}$')
#     plt.savefig(os.path.join(path_images, 'error_2.png'), bbox_inches='tight', transparent=True)
#
#     plt.figure()
#     plt.errorbar(σ_phs/pi*180, (np.mean(ε_fail, axis=1)-ε)/ε, np.ptp(ε_fail, axis=1)/2/ε, fmt='-ro')
#     plt.errorbar(σ_phs/pi*180, (np.mean(ε_mag, axis=1)-ε)/ε, np.ptp(ε_mag, axis=1)/2/ε, fmt='-bo')
#     plt.errorbar(σ_phs/pi*180, (np.mean(ε_phs, axis=1)-ε)/ε, np.ptp(ε_phs, axis=1)/2/ε, fmt='-go')
#     plt.title('$\epsilon_{o} = %.1f$'%ε)
#     plt.xlim([0, np.max(σ_phs)/pi*180])
#     plt.xlabel('σ [deg]')
#     plt.ylim([0, 2])
#     plt.gca().set_ylim(bottom=0)
#     plt.ylabel('$\Delta \epsilon / \epsilon_{o}$')
#     plt.savefig(os.path.join(path_images, 'error_3.png'), bbox_inches='tight', transparent=True)

# Plot currents.
plt.figure()
plt.scatter(x_n/λ, y_n/λ, c=np.abs(i)*1000, cmap='OrRd', vmin=0)
plt.xlabel('x [λ]')
plt.ylabel('y [λ]')
plt.gca().set_aspect('equal')
cbar = plt.colorbar()
cbar.ax.set_title('mA')
if save_option:
    plt.savefig(os.path.join(path_images, 'currents_mag.png'), bbox_inches='tight', transparent=True)

plt.figure()
plt.scatter(x_n/λ, y_n/λ, c=np.angle(i)/pi*180, cmap='OrRd', vmin=-180, vmax=180)
plt.xlabel('x [λ]')
plt.ylabel('y [λ]')
plt.gca().set_aspect('equal')
cbar = plt.colorbar(ticks=[-180, -90, 0, 90, 180])
cbar.ax.set_title('deg')
plt.set_cmap('twilight_shifted')
if save_option:
    plt.savefig(os.path.join(path_images, 'currents_phs.png'), bbox_inches='tight', transparent=True)

# Plot.
x, y = np.meshgrid(x, y)

pattern_db = 20*np.log10(np.abs(pattern))
pattern_db[pattern_db > 0] = 0
pattern_db[pattern_db < -30] = -30

solution = Z@i
solution_db = 20*np.log10(np.abs(solution))
solution_db[solution_db > 0] = 0
solution_db[solution_db < -30] = -30

levels_mag = np.linspace(-30, 0, 11)
levels_phs = np.linspace(-180, 180, 9)

# Plot fields.
# plt.figure()
# plt.contourf(x/λ, y/λ, pattern_db, levels=levels_mag)
# # plt.plot(x_o/λ, y_o/λ, 'k+')
# plt.xlabel('x [λ]')
# plt.ylabel('y [λ]')
# plt.gca().set_aspect('equal')
# plt.set_cmap('jet')
# cbar = plt.colorbar()
# cbar.ax.set_title('dBV/m')
# if save_option:
#     plt.savefig(os.path.join(path_images, 'pattern_mag.png'), bbox_inches='tight', transparent=True)

plt.figure()
plt.tricontourf(x_o/λ, y_o/λ, solution_db, levels=levels_mag)
# plt.plot(x_o/λ, y_o/λ, 'k+')
plt.xlabel('x [λ]')
plt.ylabel('y [λ]')
plt.gca().set_aspect('equal')
plt.set_cmap('seismic')
cbar = plt.colorbar()
cbar.ax.set_title('dBV/m')
if save_option:
    plt.savefig(os.path.join(path_images, 'solution_mag.png'), bbox_inches='tight', transparent=True)

# plt.figure()
# plt.contourf(x/λ, y/λ, np.angle(pattern)/pi*180, vmin=-180, vmax=180, levels=levels_phs)
# # plt.plot(x_o/λ, y_o/λ, 'k+')
# plt.xlabel('x [λ]')
# plt.ylabel('y [λ]')
# plt.gca().set_aspect('equal')
# plt.set_cmap('twilight_shifted')
# cbar = plt.colorbar(ticks=[-180, -90, 0, 90, 180])
# cbar.ax.set_title('deg')
# if save_option:
#     plt.savefig(os.path.join(path_images, 'pattern_phs.png'), bbox_inches='tight', transparent=True)

plt.figure()
plt.tricontourf(x_o/λ, y_o/λ, np.angle(solution)/pi*180, vmin=-180, vmax=180, levels=levels_phs)
# plt.plot(x_o/λ, y_o/λ, 'k+')
plt.xlabel('x [λ]')
plt.ylabel('y [λ]')
plt.gca().set_aspect('equal')
plt.set_cmap('twilight_shifted')
cbar = plt.colorbar(ticks=[-180, -90, 0, 90, 180])
cbar.ax.set_title('deg')
if save_option:
    plt.savefig(os.path.join(path_images, 'solution_phs.png'), bbox_inches='tight', transparent=True)

# Extended solution.
if save_option:
    x_ext = np.linspace(x_min, x_max, s_p_λ*n_λ_x+1)
    y_ext = np.linspace(y_min, y_max, s_p_λ*n_λ_x+1)
    z_ext = np.linspace(z_min, z_max, s_p_λ*n_λ_x+1)

    x_ext, y_ext, z_ext = np.meshgrid(x_ext, y_ext, z_ext)

    ρ_ext, θ_ext, ϕ_ext = cart2sph(x_ext, y_ext, z_ext)

    f = F@i
    solution_ext = -1j*k*shws(f, k*ρ_ext, θ_ext, ϕ_ext)

    # ATTO.
    solution_ext = np.flip(solution_ext, axis=2)

    data = np.zeros((x_ext.size, 4), dtype=float)
    data[:, 0] = x_ext.flatten()
    data[:, 1] = y_ext.flatten()
    data[:, 2] = z_ext.flatten()
    data[:, 3] = np.abs(solution_ext.flatten())
    np.savetxt(os.path.join(path_images, 'data.txt'), data)

    # Mag2dB.
    solution_ext_db = 20*np.log10(np.abs(solution_ext))
    solution_ext_db[solution_ext_db > 0] = 0
    solution_ext_db[solution_ext_db < -30] = -30

    # PyVista.
    grid = pv.StructuredGrid(x_ext, y_ext, z_ext)
    grid.dimensions = solution_ext_db.shape
    grid.point_data["values"] = solution_ext_db.flatten(order="F")
    slices = grid.slice_orthogonal(x=0, y=0, z=z)

    plotter = pv.Plotter(notebook=False, off_screen=True)
    plotter.add_mesh(slices, cmap='twilight_shifted', lighting=False, opacity=.99)
    # plotter.add_title('y = %i' %γ)
    plotter.camera.elevation = -20
    plotter.open_gif(os.path.join(path_images, 'orbit.gif'))
    for phase in np.linspace(0, 360, 181):
        plotter.camera.azimuth = phase
        plotter.write_frame()

print('Finished')
