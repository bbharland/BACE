import numpy as np
"""
Testing some change to a file in git project
"""
def calc_occam_terms_Mackay(C):
    "Return log(OF), log(sigma_T), log(sigma_{T|C}), log(P(C|L)), log(P(C|TL))"
    T = calc_T_max_L(C)
    log_evidence = calc_log_evidence_Bowman(C)
    log_likelihood = calc_log_likelihood(C, T)
    log_OF = calc_log_OF(log_evidence, log_likelihood)
    log_sigma_T = calc_log_sigma_T_uniform(T)
    # log_sigma_T = calc_log_sigma_T_maxprior(T)
    log_sigma_TC = calc_log_sigma_TC(log_OF, log_sigma_T)
    return log_OF, log_sigma_T, log_sigma_TC, log_evidence, log_likelihood

def calc_occam_terms_uniform(C):
    "Return log(OF), log(sigma_T), log(sigma_{T|C}), log(P(C|L)), log(P(C|TL))"
    T = calc_T_max_L(C)
    log_evidence = calc_log_evidence_uniform(C)
    log_likelihood = calc_log_likelihood(C, T)
    log_OF = calc_log_OF(log_evidence, log_likelihood)
    log_sigma_T = calc_log_sigma_T_uniform(T)
    log_sigma_TC = calc_log_sigma_TC(log_OF, log_sigma_T)
    return log_OF, log_sigma_T, log_sigma_TC, log_evidence, log_likelihood

def calc_occam_terms_Bowman(C):
    "Return log(OF), log(sigma_T), log(sigma_{T|C}), log(P(C|L)), log(P(C|TL))"
    T = calc_T_max_J(C)
    log_evidence = calc_log_evidence_Bowman(C)
    log_likelihood = calc_log_likelihood(C, T)
    log_OF = calc_log_OF(log_evidence, log_likelihood)
    log_sigma_T = calc_log_sigma_T_uniform(T)
    # log_sigma_T = calc_log_sigma_T_maxprior(T)
    log_sigma_TC = calc_log_sigma_TC(log_OF, log_sigma_T)
    return log_OF, log_sigma_T, log_sigma_TC, log_evidence, log_likelihood

#==========================================================================

def calc_T_max_L(C):
    "Calculate T that maximizes likelihood, P(C|TL)"
    Ci = C.sum(axis=1)
    return C / Ci.reshape(C.shape[0], 1)

def calc_T_max_J(C):
    "Calculate T that maximizes joint probability, P(CT|L)"
    n = C.shape[0]
    Ci = C.sum(axis=1)
    return (C + 1/n - 1) / (Ci.reshape(n, 1) + 1 - n)

def calc_log(A):
    with np.errstate(divide='ignore', invalid='ignore'):
        logA = np.log(A)
        logA[np.isneginf(logA) | np.isnan(logA)] = 0
    return logA

def calc_log_evidence_Bowman(C):
    n = C.shape[0]
    T = calc_T_max_L(C)
    return np.sum(C * calc_log(T)) - n**2 * np.log(n)

def calc_log_evidence_uniform(C):
    n = C.shape[0]
    Ci = C.sum(axis=1) + n - 1
    term_sumCij = np.sum(C * calc_log(C))
    term_sumCi = np.sum(Ci * calc_log(Ci))
    term_logn = n * (n-1) * np.log(n-1)
    return term_sumCij - term_sumCi + term_logn

def calc_log_likelihood(C, T):
    return np.sum(C * calc_log(T))

def calc_log_OF(log_evidence, log_likelihood):
    return log_evidence - log_likelihood

def calc_log_sigma_T_uniform(T):
    n = T.shape[0]
    return -n * (n-1) * (np.log(n-1) - 1)

def calc_log_sigma_T_maxprior(T):
    n = T.shape[0]
    return n**2 * np.log(n) + (1 - 1/n) * np.sum(calc_log(T))

def calc_log_sigma_TC(log_OF, log_sigma_T):
    return log_OF + log_sigma_T


