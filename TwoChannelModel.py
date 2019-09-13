import numpy as np
from BACEModel import BACEModel
from MarkovModel import MarkovModel

class TwoChannelModel(BACEModel):
    """
    A model with an energetic staircase of states with two channels with no 
    transitions across.

    * this model uses MarkovModel to construct model *
      -> list of potential well energies (V_i): 
                       V = [V_0, ...]
      -> table of transition barriers (E_ij)
                       E = [E, [(list of tuples)], ...]
    """
    def __init__(self, figdir='./', with_grid=True):
        "This calls self._init_C_matrix()"
        self.with_grid = with_grid
        super().__init__(figdir=figdir)
        
    def calc_log_BF_matrix(self):
        super().calc_log_BF_matrix()
        
    def merge_best_pair_states(self):
        super().merge_best_pair_states()

    def draw_graph_cbf(self, title='tmp.png'):
        super().draw_graph_cbf(edge_factor=0.02, title=title)

    def _init_C_matrix(self):
        """
        Choose to use V list, E table to fill C matrix
        """
        if self.with_grid:
            mm = MarkovModel(*self._build_V_E_with_grid(), beta=1, tau=1)
        else:
            mm = MarkovModel(*self._build_V_E_with_crosses(), beta=1, tau=1)
        mm.draw_graph()
        mm.fill_C_matrix_mu_T(nsteps=50000)
        self.C = mm.C
        
    def _build_V_E_with_crosses(self):    
        """
        Model specified here!
        V = list of potential energies (nodes)
        E = table of barrier energies (edges)
        """
        from EdgeBuilder import EdgeBuilder
 
        # edges have diagonal connections
        eb = EdgeBuilder()
        # E_ij = 3.5
        zigzag = [0, 7, 1, 0, 6, 1, 6, 2, 1, 5, 2, 5, 3, 2, 4, 3]
        e35 = eb.new_edges_series(zigzag)
        # E_ij = 3.0
        zigzag = [4, 11, 5, 10, 4, 5, 6, 9, 7, 6, 8, 7]
        e30 = eb.new_edges_series(zigzag)
        # E_ij = 2.5
        top = [10, 11, 12, 10, 13, 11]
        bottom = [8, 9, 14, 8, 15, 9]
        e25 = eb.new_edges_series(top)
        e25.extend(eb.new_edges_series(bottom))
        # E_ij = 2.0
        top = [13, 12, 19, 13, 18, 12]
        bottom = [15, 14, 17, 15, 16, 14]
        e20 = eb.new_edges_series(top)
        e20.extend(eb.new_edges_series(bottom))
        # E_ij = 1.5
        zigzag = [16, 23, 17, 16, 22, 17, 22, 18, 17, 21, 18, 21, 19,
                  18, 20, 19]
        e15 = eb.new_edges_series(zigzag)
        # E_ij = 1.0
        line = [20, 21, 22, 23]
        e10 = eb.new_edges_series(line)
        eb.draw_graph()
        
        V = [2.5] * 4 + [2] * 4 + [1.5] * 4 + [1] * 4 + [0.5] * 4 + [0] * 4
        E = [[3.5, e35], [3.0, e30], [2.5, e25], [2.0, e20],
             [1.5, e15], [1.0, e10]]
        return V, E

    def _build_V_E_with_grid(self):   
        """
        Model specified here!
        V = list of potential energies (nodes)
        E = table of barrier energies (edges)
        """ 
        from EdgeBuilder import EdgeBuilder
        # edges have diagonal connections
        eb = EdgeBuilder()
        # E_ij = 3.5
        e35 = eb.new_edges_series([0, 1, 2, 3])
        e35.extend(eb.new_edges_pairs([[0, 7], [1, 6], [2, 5], [3, 4]]))
        # E_ij = 3.0
        e30 = eb.new_edges_series([4, 5, 6, 7])
        e30.extend(eb.new_edges_pairs([[4, 11], [5, 10], [6, 9], [7, 8]]))
        # E_ij = 2.5
        e25 = eb.new_edges_series([12, 11, 10, 13])
        e25.extend(eb.new_edges_series([14, 9, 8, 15]))
        # E_ij = 2.0
        e20 = eb.new_edges_series([19, 12, 13, 18])
        e20.extend(eb.new_edges_series([17, 14, 15, 16]))
        # E_ij = 1.5
        e15 = eb.new_edges_series([20, 19, 18, 17, 16, 23])
        e15.extend(eb.new_edges_pairs([[18, 21], [17, 22]]))
        # E_ij = 1.0
        e10 = eb.new_edges_series([20, 21, 22, 23])
        eb.draw_graph()
        
        V = [2.5] * 4 + [2] * 4 + [1.5] * 4 + [1] * 4 + [0.5] * 4 + [0] * 4
        E = [[3.5, e35], [3.0, e30], [2.5, e25], [2.0, e20],
             [1.5, e15], [1.0, e10]]
        return V, E
