import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# from LogBF import LogBF

class BACEModel(object):
    """
    The base class for doing BACE clustering
    
        state_list : keep track of which states are being merged
            - initially: state_list = [[0], [1], [2], [3], [4], [5], ...]
            - later on:  state_list = [[0, 1], [2], [3, 4, 5], ...]
     
    Methods:
        __init__(figdir='./')             calls _init_C_matrix()!
        _init_C_matrix()                  define in child class!
        _init_C_from_node_edge_table(node_table, edge_table)

        calc_log_BF_matrix()              calc log(BF) for all state pairs
        merge_states(alpha)               merge states and update state_list
        merge_best_pair_states()   
        _merge_states_C(alpha)
        _merge_state_labels(alpha)

        draw_graph(edge_factor, edge_matrix, title)
        draw_graph_cbf(edge_factor, title='tmp.png')
        _build_graph(weight_matrix)
    """
    # C = None               # Counts matrix.  Indices match state_list
    # N = None               # total number of transitions in C
    # log_BF_matrix = None   # upper-triangular matrix, log(BF) for state pairs
    # state_list = None      # keep track of states being merged
    # figdir = None          # directory to put figures
    
    #=====================================================================
    #
    #   Initialization methods
    #
    #=====================================================================
    
    def __init__(self, figdir='./'):
        """
        All child classes use this __init__
        Derived classes are differentiated by _init_C_matrix() only!
        """
        if figdir[-1] == '/':
            self.figdir = figdir
        else:
            self.figdir = figdir + '/'
        
        self._init_C_matrix()
        self.N = self.C.sum()
        self.state_list = [[n] for n in range(self.C.shape[0])]
    
    def _init_C_matrix(self):
        raise RuntimeError('Trying to run BACEModel._init_C_matrix()')
        
    def _init_C_from_node_edge_table(self, node_table, edge_table):
        """
        Use node_table and edge_table to fill C matrix

        TODO: make node_table and edge_table same format
        """
        self.C = np.zeros((len(node_table), len(node_table)), dtype=int)
        for i, count in node_table:
            self.C[i, i] = count
        for count, edges in edge_table:
            for (i, j) in edges:
                self.C[i, j] = count
                self.C[j, i] = count
        
    #=====================================================================
    #
    #   Methods for merging pairs of states together
    #
    #=====================================================================
                
    def calc_log_BF_matrix(self):
        """
        Calculate (upper-triangular) matrix of log(BF)'s
        - do this only for pairs of states that are connected (C_ij > 0)
        """
        from itertools import combinations
    
        n = self.C.shape[0]
        P = self.C / self.N
        p = P.sum(axis=1)
        deltaF = self._calc_delta_F(n)
        self.log_BF_matrix = np.zeros((n, n))
        
        for i, j in combinations(range(n), 2):
            if P[i, j] > 0:
                not_ij = [k for k in range(n) if k not in (i, j)]
                a = np.array([i, j])
                na = np.array(not_ij)

                P_aa = P[a[:, np.newaxis], a]
                P_an = P[a[:, np.newaxis], na]
                Q_aa = P_aa.sum()
                Q_an = P_an.sum(axis=0)

                divpq = np.sum(p[a] * np.log(p[a] / p[a].sum()))
                divPQ = np.sum(P_aa * np.log(P_aa / Q_aa))
                with np.errstate(divide='ignore', invalid='ignore'):
                    PlogP = P_an * np.log(P_an / Q_an)
                    PlogP[np.isneginf(PlogP) | np.isnan(PlogP)] = 0
                divPQ += 2 * PlogP.sum()
                
                self.log_BF_matrix[i, j] = deltaF + divPQ - divpq

    def _calc_delta_F(self, n):
        return self._calc_F(n) - self._calc_F(n-1)

    def _calc_F(self, n):
        return -n*n*np.log(n) / self.N

    def _calc_delta_F_prior(self, n):
        return self._calc_F_prior(n) - self._calc_F_prior(n-1)

    def _calc_F_prior(self, n):
        # return (1 - (n*n / self.N)) * np.log(n)

    def merge_states(self, alpha):
        """
        1. Merge states (alpha, list) together in the C matrix
        2. Update state_list
        """
        self._merge_states_C(alpha)
        self._merge_state_labels(alpha)
    
    def merge_best_pair_states(self):
        """
        1. The log(BF) matrix must already be computed!
        2. Select the pair of states with the lowest log(BF)
        3. Merge them together in the C matrix
        4. Update state_list
        """
        if self.log_BF_matrix is None:
            raise RuntimeError('log(BF) matrix is not calculated')

        alpha = list(np.unravel_index(self.log_BF_matrix.argmin(), 
                                      self.log_BF_matrix.shape))
        self._merge_states_C(alpha)
        self._merge_state_labels(alpha)
        self.log_BF_matrix = None
    
    def _merge_states_C(self, alpha):
        """
        Update the C matrix by merging together all states in alpha
        """
        not_alpha = [i for i in range(self.C.shape[0]) if i not in alpha]
        a = np.array(alpha)
        na = np.array(not_alpha)
        
        C_aj = self.C[a, :].sum(axis=0)
        C_new = np.vstack((C_aj, self.C[na, :]))
        C_ia = C_new[:, a]
        C_ia = C_ia.sum(axis=1).reshape(C_new.shape[0], 1)
        C_new = np.hstack((C_ia, C_new[:, na]))
        self.C = C_new
    
    def _merge_state_labels(self, alpha):
        """
        Update state_labels, merging together state indices in alpha
        """
        # gather up state labels in the locations indicated by 'alpha'
        alpha_states = []
        for state_index in alpha:
            alpha_states.extend(self.state_list[state_index])

        # delete all of these locations in 'state_list' in reverse order
        alpha.sort(reverse=True)
        for state_index in alpha:
            del self.state_list[state_index]

        # prepend the set of alpha state labels at start of 'state_list'
        self.state_list.insert(0, alpha_states)
        
    #=====================================================================
    #
    #   Visualization with networkx
    #
    #===================================================================== 

    def draw_graph(self, edge_factor, edge_matrix, title):
        """
        Use networkx to render a plot of our graph showing connectivity with
        edges showing contents of edge_matrix with line thickness scaled
        by edge_factor
        """
        labels, G = self._build_graph(weight_matrix=(edge_matrix * edge_factor))
        weights = [G[i][j]['weight'] for i, j in G.edges()]

        pos = nx.spring_layout(G)
        _ = nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                   node_size=1000, alpha=1)
        _ = nx.draw_networkx_labels(G, pos, labels,
                                    font_size=12, alpha=1)
        _ = nx.draw_networkx_edges(G, pos, edgelist=G.edges, alpha=0.5,
                                   width=weights, edge_color='black')
        plt.axis('off')
        plt.savefig(self.figdir + title, dpi=200) 
        plt.show()

    def draw_graph_ax(self, ax, edge_factor, edge_matrix):
        """
        Draw graph to supplied axes object
        """
        labels, G = self._build_graph(weight_matrix=(edge_matrix * edge_factor))
        weights = [G[i][j]['weight'] for i, j in G.edges()]

        pos = nx.spring_layout(G)
        _ = nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                   node_size=1000, alpha=1, ax=ax)
        _ = nx.draw_networkx_labels(G, pos, labels,
                                    font_size=12, alpha=1, ax=ax)
        _ = nx.draw_networkx_edges(G, pos, edgelist=G.edges, alpha=0.5,
                                   width=weights, edge_color='black', ax=ax)
        ax.axis('off')

    def draw_graph_cbf(self, edge_factor, title='tmp.png'):
        """
        Draw graph with:
         - nodes : labelled according to original state labels combined
         - edges : thickness = counts * edge_factor
                   colour = cmap according to log(BF)
        """
        labels, G = self._build_graph(self.C)

        # edge line thickness = C_ij * edge_factor
        weights = [G[i][j]['weight'] * edge_factor for i, j in G.edges()]

        # edge line colour = cmap with log(BF) value
        label_list = list(labels)
        bfvals = []
        for ilabel, jlabel in G.edges:
            i = label_list.index(ilabel)
            j = label_list.index(jlabel)
            bfvals.append(self.log_BF_matrix[i, j])

        # draw graph
        pos = nx.spring_layout(G, iterations=300)
        _ = nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                   node_size=1000, alpha=1)
        _ = nx.draw_networkx_labels(G, pos, labels,
                                    font_size=12, alpha=1)
        edges = nx.draw_networkx_edges(G, pos, edgelist=G.edges, 
                                       alpha=0.8, width=weights, 
                                       edge_color=bfvals, 
                                       edge_cmap=plt.cm.viridis)
        plt.axis('off')
        plt.colorbar(edges)
        plt.savefig(self.figdir + title, dpi=200) 
        plt.show()         
        
    def _build_graph(self, weight_matrix):
        """
        Parameters:
            weight_matrix : an upper triangular matrix containing weights
                            - indexing corresponds to node indices
                            - only positive weights are added as edges!
        Return:
            labels : a dictionary of node labels used by networkx draw methods
            G : a networkx graph with edge weights given by weight_matrix
        TODO: use combinations for loops over i, j
        """
        node_list = [','.join([str(n) for n in s]) for s in self.state_list]
        labels = {node: node for node in node_list}

        G = nx.Graph()
        for node in node_list:
            G.add_node(node)

        # add edges with weights
        for i in range(len(node_list)):
            for j in range(i+1, len(node_list)):
                if weight_matrix[i, j] > 0:
                    G.add_edge(node_list[i], node_list[j], 
                               weight=weight_matrix[i, j])
        return labels, G






    
