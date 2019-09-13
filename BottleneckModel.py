import numpy as np
from BACEModel import BACEModel
from EdgeBuilder import EdgeBuilder
from MarkovModel import MarkovModel

class BottleneckModel(BACEModel):
    """
    A model with 6 microstates on either side of a central bottleneck microstate

    * this model uses MarkovModel to construct model *
      -> list of potential well energies (V_i): 
                       V = [V_0, ...]
      -> table of transition barriers (E_ij)
                       E = [E, [(list of tuples)], ...]
    """
    def __init__(self, figdir='./'):
        "This calls self._init_C_matrix()"
        super().__init__(figdir=figdir)
        
    def calc_log_BF_matrix(self):
        super().calc_log_BF_matrix()
        
    def merge_best_pair_states(self):
        super().merge_best_pair_states()

    def draw_graph_cbf(self, title='tmp.png'):
        super().draw_graph_cbf(edge_factor=0.06, title=title)

    def _init_C_matrix(self):
        """
        Choose to use V list, E table to fill C matrix
        """
        mm = MarkovModel(*self._build_V_E(), beta=1, tau=0.1)
        mm.fill_C_matrix_mu_T(nsteps=10000)
        self.C = mm.C
        
    def _build_V_E(self):
        """
        Model specified here!
        V = list of potential energies (nodes)
        E = table of barrier energies (edges)
        """
        from EdgeBuilder import EdgeBuilder
 
        eb = EdgeBuilder()
        V = [0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0]
        E = []

        # left-hand states
        E.append([0.5, eb.new_edges_series([0, 1, 2])])
        E.append([1.5, eb.new_edges_series([0, 3, 4, 5, 2])])
        E.append([2.0, eb.new_edges_fully_connected([1, 0, 3, 4])])
        E.append([2.0, eb.new_edges_fully_connected([1, 4, 5, 2])])
        
        # right-hand states
        E.append([0.5, eb.new_edges_series([10, 11, 12])])
        E.append([1.5, eb.new_edges_series([10, 7, 8, 9, 12])])
        E.append([2.0, eb.new_edges_fully_connected([7, 8, 10, 11])])
        E.append([2.0, eb.new_edges_fully_connected([8, 9, 11, 12])])

        # bottleneck state
        E.append([2.5, eb.new_edges_whiskers(6, [3, 4, 5, 7, 8, 9])])
        return V, E
