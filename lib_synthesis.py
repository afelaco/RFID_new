import numpy as np
from lib_function import vector_spherical_harmonic as vsh, scalar_spherical_harmonic as ssh
from lib_function import spherical_hankel_wave as shw


def scalar_spherical_harmonics_synthesis(c, θ, ϕ):
    """
    Computes the scalar spherical harmonics synthesis from given coefficients, defined as a
    :math:`((order+1)^2 \\times d)` ndarray, where :math:`order` is the analysis order and :math:`d` are the field's dimensions.
    The field is estimated for :math:`(m \\times n)` spherical coordinates :math:`(\\theta, \\phi)`. This function
    returns a :math:`(m \\times n \\times d)` ndarray with the angular field values.

    :param  np.ndarray  c: expansion coefficients.
    :param  np.ndarray  θ: elevation coordinates in radians.
    :param  np.ndarray  ϕ: azimuth coordinates in radians.
    :param  str         mode: scalar or vector sythesis.
    :return:            angular field values.
    :rtype: np.ndarray
    """

    # Determine order of expansion.
    L = int(np.sqrt(c.shape[0])-1)

    # Synthesis.
    f = np.zeros(θ.shape+c[0, ...].shape, dtype=complex)
    if c.ndim > 1:
        c = np.expand_dims(c, axis=(1, 2))
    for l in range(L+1):
        for m in range(-l, l+1):
            f += np.squeeze(c[l*(l+1)+m, ...]*ssh(θ, ϕ, l, m)[..., None])

    return f


def vector_spherical_harmonics_synthesis(coefficients, θ, ϕ):
    """
    Computes the scalar spherical harmonics synthesis from given coefficients, defined as a
    :math:`((order+1)^2 \\times d)` ndarray, where :math:`order` is the analysis order and :math:`d` are the field's dimensions.
    The field is estimated for :math:`(m \\times n)` spherical coordinates :math:`(\\theta, \\phi)`. This function
    returns a :math:`(m \\times n \\times d)` ndarray with the angular field values.

    :param  np.ndarray  coefficients: expansion coefficients.
    :param  np.ndarray  θ: elevation coordinates in radians.
    :param  np.ndarray  ϕ: azimuth coordinates in radians.
    :param  str         mode: scalar or vector sythesis.
    :return:            angular field values.
    :rtype: np.ndarray
    """

    # Initialization.
    field = np.zeros(θ.shape+(3,), dtype=complex)

    # Determine order of expansion.
    L = int(np.sqrt(coefficients.shape[-1])-1)

    # Vector field case.
    coefficients = np.expand_dims(coefficients, axis=tuple(range(1, θ.ndim+2)))
    for l in range(L+1):
        for m in range(-l, l+1):
            field += np.sum(coefficients[..., l*(l+1)+m]*np.array(vsh(θ, ϕ, l, m)), axis=0)

    return field


def spherical_hankel_waves_synthesis(c, ρ, θ, ϕ):
    """
    Computes the scalar spherical waves synthesis from given coefficients, defined as a
    :math:`((L+1)^2 \\times d)` ndarray, where :math:`L` is the analysis order and :math:`d` are the field's dimensions.
    The field is estimated for :math:`(m \\times n \\times o)` spherical coordinates :math:`(\\rho, \\theta, \\phi)`.
    This function returns a :math:`(m \\times n \\times o \\times d)` ndarray with the 3D field values.

    :param  np.ndarray  coefficients: expansion coefficients.
    :param  np.ndarray  ρ: radial coordinates.
    :param  np.ndarray  θ: elevation coordinates in radians.
    :param  np.ndarray  ϕ: azimuth coordinates in radians.
    :return:            angular field values.
    :rtype: np.ndarray
    """

    # Determine order of expansion.
    L = int(np.sqrt(c.shape[0])-1)

    # Synthesis.
    f = np.zeros(θ.shape+c[0, ...].shape, dtype=complex)
    if c.ndim > 1:
        c = np.expand_dims(c, axis=(1, 2))
    for l in range(L+1):
        for m in range(-l, l+1):
            f += np.squeeze(c[l*(l+1)+m, ...]*shw(ρ, θ, ϕ, l, m)[..., None])

    return f
