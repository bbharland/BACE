import numpy as np
from itertools import combinations

"""
class LogBF
class LogBFExtendedStirling

Helper functions:
-----------------
// calc_C_merged(C, alpha)                             Return new C matrix
// calc_state_list_merged(state_list, alpha)           Return new C state_list
calc_logBF_table_microstates(C, bf, state_list)     [i, j, logBF_micro]
sort_table(template_table, input_table)             Return sorted table
find_table_value(table, alpha)
find_table_index(table, alpha)
get_state_label(state_list, index)
get_alpha_label(state_list, alpha)
"""

class LogBF(object):
    """
    Collection of methods for various log(P(x_N|M)), log(BF) terms
    -----------------------------------------------------------------

    calc_logBF_table                   return sorted table: [i j logBF]

    calc_logBF_evidence_macrostates    dG (macrostates), using evidence formula
    calc_log_evidence_macrostates      G (macrostates)
    calc_logBF_evidence_microstates    dG (microstates), using evidence formula
    calc_logBF_microstates             dG (microstates)
    calc_log_evidence_microstates      G (microstates)
    calc_logBF_Bowman_eq3              dG (macrostates), Bowman formula
    calc_logBF_Bowman_PQ               dG (macrostates), Bowman dGaa, PQ formula
    calc_logBF_macrostates             dG (macrostates), my dGaa, PQ formula
    calc_sum_PQ_terms                  return sumpq_a, sumPQ_aa, sumPQ_an
    _calc_p_P_blocks                   return p_a, P_aa, P_an

    calc_delta_F_Bowman
    _calc_F_Bowman                     F (macrostates), n^2 (1 - log(n))
    calc_delta_F
    _calc_F                            F (macrostates), -n^2 log(n)
    calc_delta_F_prior
    _calc_F_prior                      F (macrostates), (N - n^2) log(n)
    """
    def __init__(self, C):
        """
        N : total counts
        Ci : original microstate counts (for microstate logBF)
        """
        self.N = C.sum()
        self.Ci = C.sum(axis=1)
        
    def calc_logBF_table(self, C, state_list, Fmethod=None, Gmethod=None):
        """
        Parameters : 
           Fmethod = None    no deltaF
                     1       deltaF, Bowman
                     2       deltaF, no prior
                     3       deltaF, with prior

           Gmethod = 1       macrostates, using delta evidence
                     2       macrostates, Bowman, Eq. 3
                     3       macrostates
                     4       microstates and macrostates
                     5       microstates

        Return : sorted table with rows = [i, j, logBF]
        """
        n = C.shape[0]

        if Fmethod is None:
            dF = 0
        elif Fmethod is 1:
            dF = self.calc_delta_F_Bowman(n)
        elif Fmethod is 2:
            dF = self.calc_delta_F(n)
        elif Fmethod is 3:
            dF = self.calc_delta_F_prior(n)
        else:
            raise RuntimeError('Fmethod must be None, 1, 2, or 3')

        logBF_table = []
        for i, j in combinations(range(n), 2):
            if C[i, j] > 0:
                if Gmethod is 1:
                    f = self.calc_logBF_evidence_macrostates
                elif Gmethod is 2:
                    f = self.calc_logBF_Bowman_eq3
                elif Gmethod is 3 or Gmethod is 4:
                    f = self.calc_logBF_macrostates
                elif Gmethod is 5:
                    f = self.calc_logBF_microstates
                else:
                    raise RuntimeError('Gmethod must be 1, 2, 3, 4, or 5')
                
                dG = f(C, [i, j])
                if Gmethod is 4:
                    dG += self.calc_logBF_microstates(state_list, [i, j])
                    
                logBF_table.append([i, j, dF + dG])

        logBF_table = np.array(logBF_table)
        return logBF_table[logBF_table[:,2].argsort()]

    #===========================================================
    #    Delta G Methods
    #===========================================================

    def calc_logBF_evidence_macrostates(self, C, alpha):
        f = self.calc_log_evidence_macrostates
        return f(C) - f(calc_C_merged(C, alpha))

    def calc_log_evidence_macrostates(self, C):
        """
        Calculate Bowman's Eq. 10 (just the C_ij part)
        """
        Ci = C.sum(axis=1)
        p = C / Ci.reshape(C.shape[0], 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            Clogp = C * np.log(p)
            Clogp[np.isneginf(Clogp) | np.isnan(Clogp)] = 0
        return Clogp.sum()
        
    def calc_logBF_evidence_microstates(self, state_list, alpha):
        f = self.calc_log_evidence_microstates
        return f(state_list) - f(calc_state_list_merged(state_list, alpha))

    def calc_logBF_microstates(self, state_list, alpha):
        """
        Calculate the microstate part of Bowman's Eq. 12, 16
        (see notes, "Bowman Figures", pp. 10-11)
        """
        C_M = np.array([self.Ci[state_list[M]].sum() for M in alpha])
        return np.sum(C_M * np.log(C_M.sum() / C_M))

    def calc_log_evidence_microstates(self, state_list):
        log_evidence = 0
        for M in state_list:
            B_M = self.Ci[M].sum()
            theta_M = self.Ci[M] / B_M
            H_theta_M = -np.sum(theta_M * np.log(theta_M))
            log_evidence -= B_M * H_theta_M
        return log_evidence
    
    def calc_logBF_Bowman_eq3(self, C, alpha):
        """
        Calculate Bowman's Eq. 3 : logBF = C_i * D(p_i||q) + ..
        """
        if len(alpha) == 2: i, j = alpha
        else: raise RuntimeError('Alpha must be length two')

        Ci = C.sum(axis=1)
        p = C / Ci.reshape(C.shape[0], 1)
        q = (Ci[i]*p[i, :] + Ci[j]*p[j, :]) / (Ci[i] + Ci[j])
    
        with np.errstate(divide='ignore', invalid='ignore'):
            plogpiq = p[i, :] * np.log(p[i, :] / q)
            plogpjq = p[j, :] * np.log(p[j, :] / q)
            plogpiq[np.isneginf(plogpiq) | np.isnan(plogpiq)] = 0
            plogpjq[np.isneginf(plogpjq) | np.isnan(plogpjq)] = 0
        return Ci[i] * plogpiq.sum() + Ci[j] * plogpjq.sum()

    def calc_logBF_Bowman_PQ(self, C, alpha):
        pq, PQ_aa, PQ_an = self.calc_sum_PQ_terms(C, alpha, Bowman=True)
        return self.N * (PQ_aa + PQ_an - pq)
    
    def calc_logBF_macrostates(self, C, alpha):
        pq, PQ_aa, PQ_an = self.calc_sum_PQ_terms(C, alpha, Bowman=False)
        return self.N * (PQ_aa + 2*PQ_an - pq)
        
    def calc_sum_PQ_terms(self, C, alpha, Bowman=False):
        p_a, P_aa, P_an = self._calc_p_P_blocks(C, alpha)
        if Bowman:
            Q_aa = P_aa.sum(axis=0)
        else:
            Q_aa = P_aa.sum()
        Q_an = P_an.sum(axis=0)
        
        sumpq_a = np.sum(p_a * np.log(p_a / p_a.sum()))
        sumPQ_aa = np.sum(P_aa * np.log(P_aa / Q_aa))
        with np.errstate(divide='ignore', invalid='ignore'):
            PlogP = P_an * np.log(P_an / Q_an)
            PlogP[np.isneginf(PlogP) | np.isnan(PlogP)] = 0        
        sumPQ_an = PlogP.sum()
        return sumpq_a, sumPQ_aa, sumPQ_an
    
    def _calc_p_P_blocks(self, C, alpha):
        not_alpha = [k for k in range(C.shape[0]) if k not in alpha]
        a = np.array(alpha)
        na = np.array(not_alpha)
        P = C / self.N
        p_a = P[a, :].sum(axis=1)
        P_aa = P[a[:, np.newaxis], a]
        P_an = P[a[:, np.newaxis], na]
        return p_a, P_aa, P_an

    #===========================================================
    #    Delta F Methods
    #===========================================================

    def calc_delta_F_Bowman(self, n):
        f = self._calc_F_Bowman
        return f(n) - f(n-1)

    def _calc_F_Bowman(self, n):
        return n**2 * (1 - np.log(n))
    
    def calc_delta_F(self, n):
        f = self._calc_F
        return f(n) - f(n-1)
    
    def _calc_F(self, n):
        return -n**2 * np.log(n)

    def calc_delta_F_prior(self, n):
        f = self._calc_F_prior
        return f(n) - f(n-1)

    def _calc_F_prior(self, n):
        return (self.N - n**2) * np.log(n)


#===========================================================
#    LogBF with extended Stirling's approximation
#===========================================================


class LogBFExtendedStirling(LogBF):
    """
    Extend the accuracy of Stirling's approximation.
    
    Usual LogBF uses :   n! ~ (n/e)**n
    Here, we extend to : n! ~ sqrt(2 pi n) * (n/e)**n
    
    Since LogBF separates the evidence into F, G terms, 
    we want to just add in corrections to those functions
    
    The log(BF) terms must be computed using 
    'calc_logBF_evidence_macrostates' ONLY!
    
    self._calc_F(n)
    self.calc_delta_F(n)
    self.calc_delta_F_prior(n)
    self.calc_logBF_evidence_macrostates(C, [i, j])
    
    """
    def calc_logBF_table(self, C, state_list, Fmethod=2, Gmethod=1):
        """
        Parameters : 
           Fmethod = 2       deltaF, no prior
                     3       deltaF, with prior

           Gmethod = 1       macrostates, using delta evidence

        Return : sorted table with rows = [i, j, logBF]
        """
        n = C.shape[0]
        if Fmethod is 2:
            dF = self.calc_delta_F(n)
        elif Fmethod is 3:
            dF = self.calc_delta_F_prior(n)
        else:
            raise RuntimeError('Fmethod must be 2 or 3')

        logBF_table = []
        for i, j in combinations(range(n), 2):
            if C[i, j] > 0:
                if Gmethod is 1:
                    dG = self.calc_logBF_evidence_macrostates(C, [i, j])
                else:
                    raise RuntimeError('Gmethod must be 1')
                    
                logBF_table.append([i, j, dF + dG])

        logBF_table = np.array(logBF_table)
        return logBF_table[logBF_table[:,2].argsort()]
    
    def calc_log_evidence_macrostates(self, C):
        le = super().calc_log_evidence_macrostates(C)
            
        # correction: 1/2 sum_ij log(C_ij)
        with np.errstate(divide='ignore', invalid='ignore'):
            logC = np.log(C)
            logC[np.isneginf(logC) | np.isnan(logC)] = 0
        le += 0.5 * np.sum(logC)
        
        # correction: -1/2 sum_i log(C_i)
        le -= 0.5 * np.sum(np.log(self.Ci))
        return le
    
    def _calc_F_correction(self, n):
        return 0.5*n*(n - 1) * np.log(2*np.pi)

    def _calc_F(self, n):
        return super()._calc_F(n) + self._calc_F_correction(n)
    
    def _calc_F_prior(self, n):
        return super()._calc_F_prior(n) + self._calc_F_correction(n)
    

#===========================================================
#    Helper functions
#===========================================================


def calc_C_merged(C, alpha):
    """
    Return : new C matrix with states alpha merged together
    """
    not_alpha = [i for i in range(C.shape[0]) if i not in alpha]
    a = np.array(alpha)
    na = np.array(not_alpha)
    C_aj = C[a, :].sum(axis=0)
    C_new = np.vstack((C_aj, C[na, :]))
    C_ia = C_new[:, a]
    C_ia = C_ia.sum(axis=1).reshape(C_new.shape[0], 1)
    return np.hstack((C_ia, C_new[:, na]))
    
def calc_state_list_merged(state_list, alpha):
    """
    Return : new state_list with states alpha merged together
    """
    new_state_list = state_list.copy()
    new_alpha = alpha.copy()
    alpha_states = []
    for state_index in new_alpha:
        alpha_states.extend(state_list[state_index])
    # delete all of these locations in 'state_list' in reverse order
    new_alpha.sort(reverse=True)
    for state_index in new_alpha:
        del new_state_list[state_index]
    # prepend the set of alpha state labels at start of 'state_list'
    new_state_list.insert(0, alpha_states)
    return new_state_list

def calc_logBF_table_microstates(C, bf, state_list):
    """
    Return table: [i, j, logBF_micro]
    """
    logBF_table = []
    for i, j in combinations(range(C.shape[0]), 2):
        if C[i, j] > 0:
            logbf = bf.calc_logBF_microstates(state_list, [i, j])
            logBF_table.append([i, j, logbf])
    return np.array(logBF_table)

def sort_table(template_table, input_table):
    """
    Return input_table, sorted as template_table
    """
    sorted_table = []
    for row in template_table:
        i, j = int(row[0]), int(row[1])
        index = find_table_index(input_table, [i, j])
        sorted_table.append(input_table[index, :])
    return np.array(sorted_table)

def find_table_value(table, alpha):
    for row in table:
        i, j = int(row[0]), int(row[1])
        if [i, j] == alpha or [j, i] == alpha:
            return row[2]
        
def find_table_index(table, alpha):
    for index, row in enumerate(table):
        i, j = int(row[0]), int(row[1])
        if [i, j] == alpha or [j, i] == alpha:
            return index

def get_state_label(state_list, index):
    return ','.join([str(i) for i in state_list[index]])

def get_alpha_label(state_list, alpha):
    return '-'.join([get_state_label(state_list, s) for s in alpha]) 
