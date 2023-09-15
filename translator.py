import numpy as np
import warnings
from numpy import abs, cos, exp, sign, sum
from scipy.special import binom as bnm, factorial as fct, spherical_jn

# Ignore warnings.
warnings.filterwarnings("ignore")


def translator(ρ, θ, ϕ, Λ_min, Λ_max, L, T):
    # Check base size.
    if np.sqrt(T.shape[0])-1 < Λ_max or np.sqrt(T.shape[1])-1 < L:
        raise ValueError("Translator base is too small.")

    # Dimensions.
    M_min = (Λ_min+1)**2
    M_max = (Λ_max+1)**2
    N = (L+1)**2
    O = Λ_max+L+1

    # Order and degree vectors.
    i = np.expand_dims(np.arange(M_min, M_max), axis=(1, 2, 3))
    λ = np.int64(np.sqrt(i))
    μ = i-λ*(λ+1)

    i = np.expand_dims(np.arange(N), axis=(0, 2, 3))
    l = np.int64(np.sqrt(i))
    m = i-l*(l+1)

    q = np.expand_dims(np.arange(O), axis=(0, 1, 3))

    s = np.expand_dims(np.arange(O), axis=(0, 1, 2))

    # Matrix.
    T = T[M_min:M_max, :N, :O]

    # Associated Legendre functions (https://en.wikipedia.org/wiki/Associated_Legendre_polynomials).
    P = (-1)**abs(m-μ)*2**q*(1-cos(θ)**2)**(abs(m-μ)/2)*fct(s)/fct(s-abs(m-μ))*cos(θ)**(s-abs(m-μ))*bnm(q, s)*bnm((q+s-1)/2, q) \
        *((1+sign(m-μ))+(1-sign(m-μ))*(-1)**abs(m-μ)*fct(q-abs(m-μ))/fct(q+abs(m-μ)))/2
    P[(s < abs(m-μ)) | (s > q)] = 0
    P = sum(P, axis=-1)

    # Phase term.
    E = exp(1j*(m-μ)*ϕ)[..., 0]

    # Spherical Bessel function.
    J = np.zeros([1, 1, O], dtype=float)
    for q in range(Λ_max+L+1):
        J[..., q] = spherical_jn(q, ρ)

    # Translator.
    T = sum(T*P*E*J, axis=-1)

    return T
