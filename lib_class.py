import matplotlib.pyplot as plt
import numpy as np
import os
from lib_analysis import scalar_spherical_harmonics_analysis as ssha
from lib_analysis import vector_spherical_harmonics_analysis as vsha
from lib_synthesis import scalar_spherical_harmonics_synthesis as sshs
from lib_synthesis import vector_spherical_harmonics_synthesis as vshs
from lib_utility import cart2sph, sph2cart
from matplotlib import cm
from numpy import abs, cos, exp, log10, max, mean, sin, sqrt, sum
from scipy.constants import c, pi
from translator import translator


class Antenna:

    def __init__(self, path):
        self.far_field = FarField(np.loadtxt(os.path.join(path, 'far_field.txt')))
        self.phase_center = None
        # self.p = np.loadtxt(os.path.join(path, 'polarization.txt'), dtype=complex)
        # self.p /= np.linalg.norm(self.p)
        self.f = float(open(os.path.join(path, 'frequency.txt')).read())
        self.v = float(open(os.path.join(path, 'voltage.txt')).read())
        self.z = complex(open(os.path.join(path, 'impedance.txt')).read())
        self.λ = c/self.f
        self.k = 2*pi/self.λ
        self.sft = None

    def spatial_fourier_transform(self, mode):
        self.sft = SpatialFourierTransform(self.far_field.field, self.far_field.θ, self.far_field.ϕ, mode)

    def translate(self, r, Λ, base):
        x, y, z = r
        ρ, θ, ϕ = cart2sph(x, y, z)

        if ρ > 0:
            self.sft.translate(self.k*ρ, θ, ϕ, Λ, base)
            self.phase_center = self.phase_center+r

    def update(self, order):
        self.sft.update(self.far_field, order)


class FarField:

    def __init__(self, far_field_data):

        self.basis = 'polar'

        n_θ = np.unique(far_field_data[:, 0]).size
        n_ϕ = np.unique(far_field_data[:, 1]).size

        self.field = np.zeros((n_θ, n_ϕ, 2), dtype=complex)

        self.field[:, :, 0] = np.reshape(far_field_data[:, 2]*exp(1j*np.radians(far_field_data[:, 3])), (n_θ, n_ϕ), order='F')
        self.field[:, :, 1] = np.reshape(far_field_data[:, 4]*exp(1j*np.radians(far_field_data[:, 5])), (n_θ, n_ϕ), order='F')

        self.θ, self.ϕ = np.radians(np.meshgrid(np.unique(far_field_data[:, 0]), np.unique(far_field_data[:, 1]), indexing='ij'))

    def change_basis(self, new_basis):

        if self.basis == 'cartesian' and new_basis == 'polar':
            field = np.zeros(self.θ.shape+(2,), dtype=complex)

            field[..., 0] = cos(self.θ)*cos(self.ϕ)*self.field[..., 0]+cos(self.θ)*sin(self.ϕ)*self.field[..., 1]-sin(self.θ)*self.field[..., 2]
            field[..., 1] = cos(self.ϕ)*self.field[..., 1]-sin(self.ϕ)*self.field[..., 0]

            self.basis = new_basis
            self.field = field

        elif self.basis == 'cartesian' and new_basis == 'spherical':
            field = np.zeros(self.θ.shape+(3,), dtype=complex)

            field[..., 0] = 1/sqrt(2)*(1j*self.field[..., 1]-self.field[..., 0])
            field[..., 1] = 1/sqrt(2)*(self.field[..., 0]+1j*self.field[..., 1])
            field[..., 2] = self.field[..., 2]

            self.basis = new_basis
            self.field = field

        elif self.basis == 'polar' and new_basis == 'cartesian':
            field = np.zeros(self.θ.shape+(3,), dtype=complex)

            field[..., 0] = cos(self.θ)*cos(self.ϕ)*self.field[..., 0]-sin(self.ϕ)*self.field[..., 1]
            field[..., 1] = cos(self.θ)*sin(self.ϕ)*self.field[..., 0]+cos(self.ϕ)*self.field[..., 1]
            field[..., 2] = -sin(self.θ)*self.field[..., 0]

            self.basis = new_basis
            self.field = field

        elif self.basis == 'polar' and new_basis == 'spherical':
            field = np.zeros(self.θ.shape+(3,), dtype=complex)

            field[..., 0] = exp(-1j*self.ϕ)/sqrt(2)*(1j*self.field[..., 1]-cos(self.θ)*self.field[..., 0])
            field[..., 1] = exp(1j*self.ϕ)/sqrt(2)*(1j*self.field[..., 1]+cos(self.θ)*self.field[..., 0])
            field[..., 2] = -sin(self.θ)*self.field[..., 0]

            self.basis = new_basis
            self.field = field

        elif self.basis == 'spherical' and new_basis == 'cartesian':
            field = np.zeros(self.θ.shape+(3,), dtype=complex)

            field[..., 0] = 1/sqrt(2)*(self.field[..., 1]-self.field[..., 0])
            field[..., 1] = -1j/sqrt(2)*(self.field[..., 0]+self.field[..., 1])
            field[..., 2] = self.field[..., 2]

            self.basis = new_basis
            self.field = field

        elif self.basis == 'spherical' and new_basis == 'polar':
            field = np.zeros(self.θ.shape+(2,), dtype=complex)

            field[..., 0] = 1/sqrt(2)*(
                    cos(self.θ)*(exp(-1j*self.ϕ)*field[..., 1]-exp(1j*self.ϕ)*field[..., 0])-sin(self.θ)*field[..., 2])
            field[..., 1] = -1j*(exp(1j*self.ϕ)*field[..., 0]+exp(-1j*self.ϕ)*field[..., 1])

            self.basis = new_basis
            self.field = field

    def show(self):

        ρ = np.linalg.norm(self.field, axis=-1)
        ρ = ρ/max(ρ)
        x, y, z = sph2cart(ρ, self.θ, self.ϕ)

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, z, facecolors=cm.jet(ρ))
        ax.set_aspect('equal')
        ax.quiver(0, 0, 0, 1.5, 0, 0)
        ax.quiver(0, 0, 0, 0, 1.5, 0)
        ax.quiver(0, 0, 0, 0, 0, 1.5)
        plt.show()


class SpatialFourierTransform:

    def __init__(self, field, θ, ϕ, mode):

        self.accuracy = 0
        self.L = -1

        while self.accuracy > -30:
            self.L += 1
            if mode == 'scalar':
                self.coefficients = ssha(field, θ, ϕ, self.L)
                self.accuracy = max(20*np.log10(np.abs(field-sshs(self.coefficients, θ, ϕ))))
            elif mode == 'vector':
                self.coefficients = vsha(field, θ, ϕ, self.L)
                self.accuracy = max(20*np.log10(np.abs(field-vshs(self.coefficients, θ, ϕ))))

            print('order =', self.L, '/ accuracy =', round(self.accuracy, ndigits=1), 'dB', end='\r')

        self.power = sum(np.linalg.norm(self.coefficients, axis=-1)**2)

    def translate(self, ρ, θ, ϕ, Λ, base):

        if Λ == -1:
            coefficients = np.empty([0, 3], dtype=complex)
            ratio = 0
            while ratio < 0.99:
                Λ += 1
                T = translator(ρ, θ, ϕ, Λ-1, Λ, self.L, base)
                coefficients = np.concatenate((coefficients, T@self.coefficients), axis=0)
                power = sum(np.linalg.norm(coefficients, axis=1)**2)
                ratio = power/self.power

                print('order =', Λ, '/ ratio =', round(ratio*100, ndigits=1), '%', end='\r')

            self.L = Λ
            self.coefficients = coefficients
            self.power = power

        else:
            T = translator(ρ, θ, ϕ, -1, Λ, self.L, base)
            self.L = Λ
            self.coefficients = T@self.coefficients
            self.power = sum(np.linalg.norm(self.coefficients, axis=-1)**2)

    def rotate(self, α, β, γ):
        """ Extrinsic rotation by Euler angles α, β, γ, about axes x, y, z. """

        R_x = np.array([[1, 0, 0], [0, cos(α), -sin(α)], [0, sin(α), cos(α)]])
        R_y = np.array([[cos(β), 0, sin(β)], [0, 1, 0], [-sin(β), 0, cos(β)]])
        R_z = np.array([[cos(γ), -sin(γ), 0], [sin(γ), cos(γ), 0], [0, 0, 1]])

        R = R_z@R_y@R_x
        self.coefficients = R@self.coefficients
