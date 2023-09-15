import numpy as np
import sparse
import warnings
from numpy import abs, squeeze, sqrt, sum
from scipy.special import factorial as fct

# Ignore warnings.
warnings.filterwarnings("ignore")

# Settings.
lib_path = 'C:\\Users\\Administrator\\OneDrive - UGent\\Python\\NFFA\\lib\\translator'

Λ = 30
L = 20

# Dimensions.
M = (Λ+1)**2
N = (L+1)**2
O = Λ+L+1

# Order and degree vectors.
i = np.expand_dims(np.arange(M), axis=(1, 2, 3))
λ = np.floor(np.sqrt(i))
μ = i-λ*(λ+1)

i = np.expand_dims(np.arange(N), axis=(0, 2, 3))
l = np.floor(np.sqrt(i))
m = i-l*(l+1)

q = np.expand_dims(np.arange(O), axis=(0, 1, 3))

s = np.expand_dims(np.arange(O), axis=(0, 1, 2))

# Normalization.
Norm = squeeze(1j**(-q)*(-1.)**(λ-l+μ)*(2*q+1)*sqrt((2*l+1)*(2*λ+1)*fct(q-m+μ)/fct(q+m-μ)))

# Delta function (https://dlmf.nist.gov/34.2.E5).
Δ = sqrt(fct(l+λ-q)*fct(l-λ+q)*fct(-l+λ+q)/fct(l+λ+q+1))

# Wigner 3j symbols (https://dlmf.nist.gov/34.2.E4).
W_1 = (-1.)**(l-λ+s)*Δ*fct(l)*fct(λ)*fct(q) \
      /fct(s)/fct(q-λ+s)/fct(q-l+s)/fct(l+λ-q-s)/fct(l-s)/fct(λ-s)
W_1[np.isnan(W_1) | np.isinf(W_1)] = 0
W_1 = sum(W_1, axis=-1)

W_2 = (-1.)**(l-λ-μ+m+s)*Δ*sqrt(fct(l+m)*fct(l-m)*fct(λ+μ)*fct(λ-μ)*fct(q+μ-m)*fct(q-μ+m)) \
      /fct(s)/fct(q-λ+m+s)/fct(q-l+μ+s)/fct(l+λ-q-s)/fct(l-m-s)/fct(λ-μ-s)
W_2[np.isnan(W_2) | np.isinf(W_2)] = 0
W_2 = sum(W_2, axis=-1)

# Precomputed translator.
T = Norm*W_1*W_2

# Boolean.
B = (abs(l-λ) <= q) & (q <= l+λ) & (abs(μ-m) <= q)
T[~B[..., 0]] = 0

# Sparse.
T = sparse.COO(T)

# Save.
sparse.save_npz(lib_path, T)
print('Update complete.')
