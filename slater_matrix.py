import numpy as np
from numpy import sqrt
from scipy.constants import pi
from scipy.special import factorial as fct


def w3j(j_1, m_1, j_2, m_2, j_3, m_3):
    """ Compute the Wigner 3j symbol. """

    # Convert int and float into np.int32 and np.float32.
    args = list(np.array(arg) for arg in [j_1, m_1, j_2, m_2, j_3, m_3])

    # Wigner 3j symbol(https://dlmf.nist.gov/34.2.E4).
    s = np.expand_dims(np.arange(0, np.max([np.max(j_1+j_2-j_3), np.max(j_1-m_1), np.max(j_2+m_2)])+1),
                       axis=tuple(range(0, max(arg.ndim for arg in args))))

    for count, arg in enumerate(args):
        if arg.ndim != 0:
            args[count] = args[count][..., None]

    j_1, m_1, j_2, m_2, j_3, m_3 = args

    # Delta function(https://dlmf.nist.gov/34.2.E5).
    d = sqrt(fct(j_1+j_2-j_3)*fct(j_1-j_2+j_3)*fct(-j_1+j_2+j_3)/fct(j_1+j_2+j_3+1))

    # Wigner 3j symbol.
    w3j = (-1.)**(j_1-j_2-m_3+s)*d*sqrt(fct(j_1+m_1)*fct(j_1-m_1)*fct(j_2+m_2)*fct(j_2-m_2)*fct(j_3+m_3)*fct(j_3-m_3)) \
          /(fct(s)*fct(j_1+j_2-j_3-s)*fct(j_1-m_1-s)*fct(j_2+m_2-s)*fct(j_3-j_2+m_1+s)*fct(j_3-j_1-m_2+s))

    # Selection rules.
    w3j[(abs(m_1) > j_1) & (abs(m_2) > j_2) & (abs(m_3) > j_3) & (w3j == w3j)] = 0
    w3j[(m_1+m_2+m_3 != 0) & (w3j == w3j)] = 0
    w3j[(j_3 < abs(j_1-j_2)) | (j_3 > j_1+j_2) & (w3j == w3j)] = 0

    # Factorial conditions.
    w3j[np.isnan(w3j) | np.isinf(w3j)] = 0

    w3j = np.sum(w3j, axis=-1)

    return w3j


def cgc(j_1, m_1, j_2, m_2, j_3, m_3):
    """ Compute the Clebsch-Gordan coefficient. """

    cgc = np.array((-1.)**(-j_1+j_2-m_3)*sqrt(2*j_3+1)*w3j(j_1, m_1, j_2, m_2, j_3, -m_3))
    cgc[np.isnan(cgc) | np.isinf(cgc)] = 0

    return cgc


def slater_matrix(L, mode):

    M = (L+1)**2
    
    i_1 = np.arange(M)[:, None, None]
    l_1 = np.floor(sqrt(i_1))
    m_1 = i_1-l_1*(l_1+1)

    i_2 = np.arange(M)[None, :, None]
    l_2 = np.floor(sqrt(i_2))
    m_2 = i_2-l_2*(l_2+1)

    i_3 = np.arange(M)[None, None, :]
    l_3 = np.floor(sqrt(i_3))
    m_3 = i_3-l_3*(l_3+1)

    if mode == 'scalar':
        slater = (-1)**l_3*sqrt((2*l_1+1)*(2*l_2+1)/(2*l_3+1)/4/pi) \
                 *cgc(l_1, 0, l_2, 0, l_3, 0)*cgc(l_1, m_1, l_2, m_2, l_3, m_3)

        np.save('lib\\ssh\\slater_matrix.npy', slater)

    elif mode == 'vector':

        l_1, m_1, l_2, m_2, l_3, m_3 = (arg[..., None] for arg in [l_1, m_1, l_2, m_2, l_3, m_3])

        m_s = np.array([-1, 0, 1])[None, None, None, :]

        slater_self = np.sum((-1)**l_3*sqrt((2*l_1+1)*(2*l_2+1)/(2*l_3+1)/4/pi)*cgc(l_1, 0, l_2, 0, l_3, 0) \
                    *cgc(l_2, m_2-m_s, 1, m_s, l_2, m_2)*cgc(l_3, m_3-m_s, 1, m_s, l_3, m_3)*cgc(l_1, m_1, l_2, m_2-m_s, l_3, m_3-m_s), axis=-1)

        slater_cross = np.sum((-1)**l_3*sqrt((2*l_1+1)*(2*l_2+1)/(2*l_3-1)*(l_3+1)/(2*l_3+1)/4/pi)*cgc(l_1, 0, l_2, 0, l_3-1, 0) \
                    *cgc(l_2, m_2-m_s, 1, m_s, l_2, m_2)*cgc(l_3-1, m_3-m_s, 1, m_s, l_3, m_3)*cgc(l_1, m_1, l_2, m_2-m_s, l_3-1, m_3-m_s) \
                    +(-1)**l_3*sqrt((2*l_1+1)*(2*l_2+1)/(2*l_3+3)*l_3/(2*l_3+1)/4/pi)*cgc(l_1, 0, l_2, 0, l_3+1, 0) \
                    *cgc(l_2, m_2-m_s, 1, m_s, l_2, m_2)*cgc(l_3+1, m_3-m_s, 1, m_s, l_3, m_3)*cgc(l_1, m_1, l_2, m_2-m_s, l_3+1, m_3-m_s), axis=-1)

        slater_self[np.isnan(slater_self) | np.isinf(slater_self)] = 0
        slater_cross[np.isnan(slater_cross) | np.isinf(slater_cross)] = 0

        np.save('lib\\vsh\\slater_self.npy', slater_self)
        np.save('lib\\vsh\\slater_cross.npy', slater_cross)
