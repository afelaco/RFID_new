import numpy as np
import os
from interpolate import interpolate
from lib_function import vector_spherical_harmonic as vsh, scalar_spherical_harmonic as ssh
from lebedev import lebedev
from scipy.constants import pi

""" A collection of analysis tools. """


def scalar_spherical_harmonics_analysis(f, θ, ϕ, L):
    """
    Compute the scalar spherical harmonics analysis of a scalar or vector angular field, defined as a
    :math:`(m \\times n \\times d)` ndarray on a :math:`(m \\times n)` regular grid in spherical coordinates, where
    :math:`d` are the field's dimensions. It returns a :math:`((L+1)^2 \\times d)` ndarray of expansion coefficients.

    :param  np.ndarray  f: values to analyse.
    :param  np.ndarray  θ: elevation coordinates in radians.
    :param  np.ndarray  ϕ: azimuth coordinates in radians.
    :param  int         L: analysis order.
    :param  str         mode: analysis mode.
    :return:            expansion coefficients.
    :rtype: np.ndarray
    """

    # Dimensions.
    M = (L+1)**2

    # Get quadrature nodes and weights.
    θ_i, ϕ_i, w, scheme = lebedev(2*L)

    # Interpolate angular field.
    f = interpolate(θ, ϕ, f, θ_i, ϕ_i)

    # Library.
    lib_path = os.path.join('lib\\ssh', scheme+'.npy')
    if os.path.isfile(lib_path):
        lib = np.load(lib_path)
    if os.path.isfile(lib_path) is False or lib.shape[-1] < M:
        lib = np.zeros(θ_i.shape+(M,), dtype=complex)
        for l in range(L+1):
            for m in range(-l, l+1):
                lib[..., l*(l+1)+m] = ssh(θ_i, ϕ_i, l, m)
        np.save(lib_path, lib)

    # Analysis.
    c = np.zeros((M,)+f[0, ...].shape, dtype=complex)
    w, lib = (np.expand_dims(arg, axis=tuple(range(θ_i.ndim, f.ndim))) for arg in [w, lib])
    for i in range(M):
        c[i, ...] = np.sum(4*pi*w*f*lib[..., i].conj(), axis=tuple(range(θ_i.ndim)))

    return c


def vector_spherical_harmonics_analysis(field, θ, ϕ, L):
    """
    Compute the scalar spherical harmonics analysis of a scalar or vector angular field, defined as a
    :math:`(m \\times n \\times d)` ndarray on a :math:`(m \\times n)` regular grid in spherical coordinates, where
    :math:`d` are the field's dimensions. It returns a :math:`((L+1)^2 \\times d)` ndarray of expansion coefficients.

    :param  np.ndarray  field: values to analyse.
    :param  np.ndarray  θ: elevation coordinates in radians.
    :param  np.ndarray  ϕ: azimuth coordinates in radians.
    :param  int         L: analysis order.
    :param  str         mode: analysis mode.
    :return:            expansion coefficients.
    :rtype: np.ndarray
    """

    M = (L+1)**2

    # Get quadrature nodes and weights.
    θ_i, ϕ_i, weights, scheme = lebedev(2*L)

    # Interpolate angular field.
    field = interpolate(θ, ϕ, field, θ_i, ϕ_i)

    # Library.
    lib_path = os.path.join('lib\\vsh', scheme+'.npy')
    if os.path.isfile(lib_path):
        lib = np.load(lib_path)
    if os.path.isfile(lib_path) is False or lib.shape[-1] < M:
        lib = np.zeros((2,)+θ_i.shape+(3, M), dtype=complex)
        for l in range(L+1):
            for m in range(-l, l+1):
                lib[..., l*(l+1)+m] = vsh(θ_i, ϕ_i, l, m)
        np.save(lib_path, lib)

    # Analysis.
    coefficients = np.sum(4*pi*weights[None, ..., None, None]*field[None, ..., None]*lib.conj(), axis=(1, 2))

    return coefficients