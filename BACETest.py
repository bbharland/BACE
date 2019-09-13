import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def calc_log_evidence_macrostates(C):
    """
    Calculate Bowman's Eq. 10 (just the C_ij part)
    """
    Ci = C.sum(axis=1)
    p = C / Ci.reshape(C.shape[0], 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        Clogp = C * np.log(p)
        Clogp[np.isneginf(Clogp) | np.isnan(Clogp)] = 0
    return Clogp.sum()

def calc_log_evidence_microstates_matrix(C):
    """
    """
    p = C / C.sum(axis=1).reshape(C.shape[0], 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        Clogp = C * np.log(p)
        Clogp[np.isneginf(Clogp) | np.isnan(Clogp)] = 0
    return Clogp

def calc_log_evidence_nstates_Bowman(C):
    n = C.shape[0]
    return n*n * (1 - np.log(n))

def calc_log_evidence_nstates_noprior(C):
    n = C.shape[0]
    return -n*n * np.log(n)

def calc_log_evidence_nstates_prior(C):
    n = C.shape[0]
    N = C.sum()
    return (N - n*n) * np.log(n)

def calc_logBF_table(C):
    logBF_table = []
    for i in range(C.shape[0]):
        for j in range(i+1, C.shape[1]):
            if C[i, j] > 0:
                logBF_table.append([i, j, calc_logBF_Bowman_eq3(C, [i, j])])
    logBF_table = np.array(logBF_table)
    return logBF_table[logBF_table[:,2].argsort()]

def merge_states(C, state_list, alpha):
    """
    Parameters :
        C             C-matrix
        state_list    [[0, 1], [2], [3, 4, 5], ...]
        alpha         list of state_list indices to merge
        
    Return:
        C_new            C-matrix with states merged
        new_state_list   new state_list with state labels merged
    """
    not_alpha = [i for i in range(C.shape[0]) if i not in alpha]
    a = np.array(alpha)
    na = np.array(not_alpha)

    # New C matrix
    C_aj = C[a, :].sum(axis=0)
    C_new = np.vstack((C_aj, C[na, :]))
    C_ia = C_new[:, a]
    C_ia = C_ia.sum(axis=1).reshape(C_new.shape[0], 1)
    C_new = np.hstack((C_ia, C_new[:, na]))
    
    # New state labels
    new_state_list = state_list.copy()
    alpha_states = []
    for state_index in alpha:
        alpha_states.extend(state_list[state_index])
    # delete all of these locations in 'state_list' in reverse order
    alpha.sort(reverse=True)
    for state_index in alpha:
        del new_state_list[state_index]
    # prepend the set of alpha state labels at start of 'state_list'
    new_state_list.insert(0, alpha_states)
    
    return C_new, new_state_list
