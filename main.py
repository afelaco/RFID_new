import matplotlib.pyplot as plt
import numpy as np
import os
from lib_class import Antenna
from lib_function import spherical_hankel_wave as shw
from lib_utility import cart2sph
from slater_matrix import slater_matrix

mode = 'vector'
basis = 'spherical'

# Import transmitter.
tx_path = 'tx\\horn'
tx = Antenna(tx_path)
tx.far_field.change_basis(basis)
tx.spatial_fourier_transform(mode)
# tx.far_field.show()

# Import receiver.
rx_path = 'rx\\qm_chest'
rx = Antenna(rx_path)
rx.far_field.change_basis(basis)
rx.spatial_fourier_transform(mode)
# rx.far_field.show()

# Update SFT order.
L = np.min([tx.sft.L, rx.sft.L])
M = (L+1)**2

tx.sft.coefficients = tx.sft.coefficients[:M, :]
rx.sft.coefficients = rx.sft.coefficients[:M, :]

# Slater library.
if mode == 'scalar':
    lib_path = 'lib\\ssh\\slater_matrix.npy'

    if os.path.isfile(lib_path):
        slater = np.load(lib_path)
        if slater.shape[0] > M:
            slater = slater[:M, :M, :M]
    if os.path.isfile(lib_path) is False or slater.shape[0] < M:
        slater_matrix(L, mode)
        slater = np.load(lib_path)

    interactions = tx.sft.coefficients.conj()@rx.sft.coefficients.T

    # plt.figure()
    # plt.imshow(abs(interactions))

    coefficients = np.sum(interactions[None, ...]*slater, axis=(1, 2))

elif mode == 'vector':
    lib_path_self = 'lib\\vsh\\slater_self.npy'
    lib_path_cross = 'lib\\vsh\\slater_cross.npy'

    if os.path.isfile(lib_path_self) is True:

        slater_self = np.load(lib_path_self)
        slater_cross = np.load(lib_path_cross)

        if slater_self.shape[0] > M:
            slater_self = slater_self[:M, :M, :M]
            slater_cross = slater_cross[:M, :M, :M]

    if os.path.isfile(lib_path_self) is False or slater_self.shape[0] < M:
        slater_matrix(L, mode)
        slater_self = np.load(lib_path_self)
        slater_cross = np.load(lib_path_cross)

    interactions_self = tx.sft.coefficients.conj()[1, :][:, None]*rx.sft.coefficients[1, :][None, :] \
                        -tx.sft.coefficients.conj()[0, :][:, None]*rx.sft.coefficients[0, :][None, :]

    interactions_cross = tx.sft.coefficients.conj()[0, :][:, None]*rx.sft.coefficients[1, :][None, :] \
                         -tx.sft.coefficients.conj()[1, :][:, None]*rx.sft.coefficients[0, :][None, :]

    # plt.figure()
    # plt.imshow(abs(interactions_self))

    # plt.figure()
    # plt.imshow(abs(interactions_cross))

    coefficients = np.sum(interactions_self[None, ...]*slater_self+interactions_cross[None, ...]*slater_cross, axis=(1, 2))

# Coefficients alt.
# i = np.arange(M)[:, None]
# l = np.floor(np.sqrt(i))
#
# θ_i, ϕ_i, w, _ = quadrature('lebedev', 2*L)
#
# f_tx = sshs(tx.sft.coefficients, θ_i, ϕ_i)
# f_rx = sshs((-1)**l*rx.sft.coefficients, θ_i, ϕ_i)
#
# coefficients_alt = np.zeros(M, dtype=complex)
# for l in range(L+1):
#     for m in range(-l, l+1):
#         coefficients_alt[l*(l+1)+m] = np.sum(4*pi*w*ssh(θ_i, ϕ_i, l, m).conj()*np.sum(f_tx.conj()*f_rx, axis=-1))
#
# plt.figure()
# plt.ion()
# plt.subplot(2, 1, 1)
# plt.plot(np.abs(coefficients), ':x')
# plt.plot(np.abs(coefficients_alt), ':o', markerfacecolor='none')
# plt.subplot(2, 1, 2)
# plt.plot(np.angle(coefficients), ':x')
# plt.plot(np.angle(coefficients_alt), ':o', markerfacecolor='none')

# Open voltage on receiver antenna.
x = tx.λ
y = tx.λ*np.linspace(-1, 1)
z = 0

x, y, z = np.meshgrid(x, y, z, indexing='ij')

ρ, θ, ϕ = cart2sph(x, y, z)

v_o = np.zeros(ρ.shape, dtype=complex)
for l in range(L+1):
    for m in range(-l, l+1):
        v_o -= 1/30*tx.v*rx.z*coefficients[l*(l+1)+m]*shw(tx.k*ρ, θ, ϕ, l, m)

z_l = 50
v_l = v_o*z_l/(rx.z+z_l)
p_l = 10*np.log10(np.abs(v_l)**2/(2*z_l))+30

plt.figure()
plt.plot(np.squeeze(y), np.squeeze(p_l))
plt.ylabel('Voltage on load [V]')
plt.grid()
plt.ylim(-75, -25)
