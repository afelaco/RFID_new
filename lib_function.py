import numpy as np
from numpy import abs, sqrt
from scipy.special import sph_harm, spherical_jn, spherical_yn
from slater_matrix import cgc as clebsch_gordan_coefficient


def scalar_spherical_harmonic(θ, ϕ, l, m):
    """
    Computes the scalar spherical harmonic :math:`Y^{m}_{l}(θ,ϕ)`.

    :param  np.ndarray  θ: elevation coordinates in radians.
    :param  np.ndarray  ϕ: azimuth coordinates in radians.
    :param  int         l: order.
    :param  int         m: degree.

    :rtype: np.ndarray
    """

    θ, ϕ = (np.array(arg) for arg in [θ, ϕ])
    if abs(m) <= l:
        y = sph_harm(m, l, ϕ, θ)
    else:
        y = np.zeros(θ.shape, dtype=complex)

    return y


def tensor_spherical_harmonic(θ, ϕ, j, l, m):
    """
    Computes the tensor spherical harmonic :math:`Y^{m}_{l}(θ,ϕ)`.

    :param  np.ndarray  θ: elevation coordinates in radians.
    :param  np.ndarray  ϕ: azimuth coordinates in radians.
    :param  int         j: order.
    :param  int         l: order.
    :param  int         m: degree.

    :rtype: np.ndarray
    """

    # Initialization.
    y = np.zeros(θ.shape+(3,), dtype=complex)

    # Tensor spherical harmonic.
    for i, m_s in enumerate([1, -1, 0]):
        y[..., i] = clebsch_gordan_coefficient(l, m-m_s, 1, m_s, j, m)*scalar_spherical_harmonic(θ, ϕ, l, m-m_s)

    return y


def vector_spherical_harmonic(θ, ϕ, l, m):
    """ Vector spherical harmonic. """

    # Initialization.
    E = np.zeros(θ.shape+(3,), dtype=complex)
    M = np.zeros(θ.shape+(3,), dtype=complex)

    # Vector spherical harmonics.
    if abs(m) <= l:
        E = sqrt((l+1)/(2*l+1))*tensor_spherical_harmonic(θ, ϕ, l, l-1, m) \
            +sqrt(l/(2*l+1))*tensor_spherical_harmonic(θ, ϕ, l, l+1, m)
        M = tensor_spherical_harmonic(θ, ϕ, l, l, m)

    return E, M


def spherical_hankel_function(ρ, l):

    h = spherical_jn(l, ρ)-1j*spherical_yn(l, ρ)

    return h


def spherical_bessel_wave(ρ, θ, ϕ, l, m):
    """ Spherical Hankel wave. """

    J = 1j**(-l)*spherical_jn(l, ρ)*scalar_spherical_harmonic(θ, ϕ, l, m)

    return J


def spherical_hankel_wave(ρ, θ, ϕ, l, m):
    """ Spherical Hankel wave. """

    H = 1j**(-l)*spherical_hankel_function(ρ, l)*scalar_spherical_harmonic(θ, ϕ, l, m)

    return H
