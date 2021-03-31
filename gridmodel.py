import numpy as np


def calc_M(m, n):
    return m * n


def calc_N(m, n, s, x, y):
    return m * n * (s + 4 * y) - 2 * y + 2 * x


def calc_F_noprior(M, N):
    return -(M**2 / N) * np.log(M)


def calc_delta_F_noprior(M, N):
    return calc_F_noprior(M, N) - calc_F_noprior(M - 1, N)


def calc_F(M, N):
    return (1 - M**2 / N) * np.log(M)


def calc_delta_F(M, N):
    return calc_F(M, N) - calc_F(M - 1, N)


def calc_plogp_alpha(s, x, y, N):
    z = s + x + 3 * y
    return 2 * z * np.log(2) / N


def calc_plogp_beta(s, x, y, N):
    z = s + x + 3 * y
    w = s + 4 * y
    return -(z * np.log(z / (z + w)) + w * np.log(w / (z + w))) / N


def calc_plogp_gamma(s, y, N):
    w = s + 4 * y
    return 2 * w * np.log(2) / N


def calc_PlogP(s, a, N):
    sa = 2 * s + 2 * a
    return 2 * (s * np.log(s / sa) + a * np.log(a / sa)) / N


calc_N_grid = np.vectorize(calc_N)
calc_delta_F_grid = np.vectorize(calc_delta_F)
calc_delta_F_noprior_grid = np.vectorize(calc_delta_F_noprior)
calc_plogp_alpha_grid = np.vectorize(calc_plogp_alpha)
calc_plogp_beta_grid = np.vectorize(calc_plogp_beta)
calc_plogp_gamma_grid = np.vectorize(calc_plogp_gamma)
calc_PlogP_grid = np.vectorize(calc_PlogP)
