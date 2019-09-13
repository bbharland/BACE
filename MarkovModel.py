import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class MarkovModel(object):
    """
    A class for building models using well potentials and barrier heights

    Provides methods:
        __init__(V, E, beta=1, tau=1):
           - calculate K from E, V, beta
           - calculate T from K, tau
           - calculate mu from T

        fill_C_matrix_MC(nsteps=10000):
           - simulate Markov Chain using T -> this populates C
           - symmetrize C: C = (C + C.T) // 2

        fill_C_matrix_mu_T(nsteps=10000):
           - C_ij = nsteps * mu_i * T_ij
           
        draw_graph(edge_factor, title):
           - nodes coloured ~ V_i
           - edges coloured ~ E_ij
    """
    # beta = None     # (float)        temperature
    # tau = None      # (float)        timestep
    # N = None        # (int)          number of timesteps
    # M = None        # (int)          number of states
    # states = None   # (list int)     list of states
    # K = None        # (MxM float)    rate constant matrix
    # T = None        # (MxM float)    transition matrix
    # C = None        # (MxM int)      counts matrix
    # mu = None       # (M float)      equilibrium distribution
    # V = None        # (list float)   potential energies (nodes)
    # E = None        # (table float)  barrier heights (edges)
    
    def __init__(self, V, E, beta=1, tau=1):
        self.beta = beta
        self.tau = tau
        self.M = len(V)
        self.V = V
        self.E = E
        self.states = list(range(self.M))
        self.K = self._calc_K_matrix()     
        self.T = self._calc_T_matrix()
        self.mu = self._calc_mu_vector()
        self.C = np.zeros((self.M, self.M))

    def fill_C_matrix_MC(self, nsteps=10000):
        """
        C matrix is populated in self._jump(state) method
            - throw away the Markov Chain
            - symmetrize the C matrix
        """
        self.N = nsteps
        chain = self._generate_chain(nsteps=nsteps, initial_state=0)
        self.C = ((self.C + self.C.T) // 2).astype(int)
        
    def fill_C_matrix_mu_T(self, nsteps=10000):
        """
        C matrix is filled with: C_ij = nsteps * mu_i * T_ij
        """
        self.N = nsteps
        self.C = (nsteps * self.mu.reshape((self.M, 1)) * self.T).astype(int)

    def _calc_K_matrix(self):
        """
        Fill K matrix with rate constants: 

            K_ij = exp(-beta * (E_ij - V_i))
            K_ii = -sum_j(K_ij)
            V_i = well potential for state i
            E_ij = barrier height connecting states i, j

        Use:
            V : list of V_i
            E : [[E_val, [list of tuples of connected states]], ...]
        """
        K = np.zeros((self.M, self.M))
        for E_item in self.E:
            Eij, pairs = E_item  # TODO: fix this
            for i, j in pairs:
                K[i, j] = np.exp(-self.beta * (Eij - self.V[i]))
                K[j, i] = np.exp(-self.beta * (Eij - self.V[j]))
        np.fill_diagonal(K, -K.sum(axis=1))
        return K
    
    def _calc_T_matrix(self):
        from scipy.linalg import expm
        return expm(self.K * self.tau)

    def _calc_mu_vector(self):
        from scipy.linalg import eig
        evals, evecs = eig(self.T, left=True, right=False)
        idx = evals.argsort()[::-1]   
        evals, evecs = evals[idx], evecs[:,idx]
        mu = evecs[:, 0]
        mu /= mu.sum()
        return mu
    
    def _generate_chain(self, nsteps, initial_state):
        chain = np.empty(nsteps+1, dtype=int)
        chain[0] = initial_state
        for step in range(nsteps):
            chain[step+1] = self._jump(chain[step])
        return chain

    def _jump(self, state_from):
        state_to = np.random.choice(self.states, p=self.T[state_from, :])
        self.C[state_from, state_to] += 1
        return state_to

    def draw_graph(self):
        """
        Draw graph with:
        - nodes : colour = cmap according to V_i
        - edges : colour = cmap according to E_ij
        """
        G = nx.Graph()
        node_colors = []
        node_labels = {}
        for node, V in enumerate(self.V):
            G.add_node(str(node))
            node_colors.append(V)
            node_labels[str(node)] = str(node)

        edge_colors = []
        for E, edges in self.E:
            for i, j in edges:
                G.add_edge(str(i), str(j), weight=1)
                edge_colors.append(E)

        all_values = edge_colors + node_colors
        vmin, vmax = min(all_values), max(all_values)
    
        pos = nx.spring_layout(G, iterations=300)
        n = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                   node_size=600, alpha=0.6,
                                   edge_cmap=plt.cm.viridis,
                                   vmin=vmin, vmax=vmax)

        _ = nx.draw_networkx_labels(G, pos, node_labels, font_size=12, alpha=1)

        e = nx.draw_networkx_edges(G, pos, edgelist=G.edges, 
                                   alpha=0.6, width=5, 
                                   edge_color=edge_colors, 
                                   edge_cmap=plt.cm.viridis,
                                   edge_vmin=vmin, edge_vmax=vmax)
        plt.colorbar(e)
        plt.axis('off')
        plt.savefig('tmp.png', dpi=200) 
