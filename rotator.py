import numpy as np
from scipy.special import factorial as fct


def rotator(α, β, γ, L):
    """Wigner-D matrix, z-y-z convention."""

    M = (L+1)**2

    i = np.expand_dims(np.arange(M), axis=(1, 2))
    λ = np.floor(np.sqrt(i))
    μ = i-λ*(λ+1)

    i = np.expand_dims(np.arange(M), axis=(0, 2))
    l = np.floor(np.sqrt(i))
    m = i-λ*(λ+1)

    s = np.expand_dims(np.arange(2*L+1), axis=(0, 1))

    d = np.sqrt(fct(λ+μ)*fct(λ-μ)*fct(l+m)*fct(l-m)) \
        *(-1)**(μ-m+s)*(np.cos(β/2))**(2*(λ-s)+m-μ)*(np.sin(β/2))**(μ-m+2*s) \
        /fct(λ+m-s)/fct(s)/fct(μ-m+s)/fct(λ-μ-s)

    d[np.isnan(d) | np.isinf(d)] = 0

    D = np.sum(np.exp(-1j*μ*α)*d*np.exp(-1j*m*γ), axis=-1)

    return D
