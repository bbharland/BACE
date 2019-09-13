import numpy as np
from BACEModel import BACEModel
from EdgeBuilder import EdgeBuilder

class BowmanModel(BACEModel):
    """
    A model with 9 microstates.  See Fig. 1 in Bowman paper.
    """
    def __init__(self, figdir='./'):
        "This calls self._init_C_matrix()"
        super().__init__(figdir=figdir)
        
    def calc_log_BF_matrix(self):
        super().calc_log_BF_matrix()
        
    def merge_best_pair_states(self):
        super().merge_best_pair_states()

    def draw_graph_counts_ax(self, ax):
        super().draw_graph_ax(ax, edge_factor=0.1, edge_matrix=self.C)
        
    def draw_graph_cbf(self, title='tmp.png'):
        super().draw_graph_cbf(edge_factor=0.1, title=title)

    def _init_C_matrix(self):
        """
        Choose to use node_table, edge_table to fill C matrix
        """
        node_table, edge_table = self._build_node_edge_table()
        super()._init_C_from_node_edge_table(node_table, edge_table)
        
    def _build_node_edge_table(self):
        """
        Model specified here!     
        """
        node_table = [[node, 1000] for node in range(9)]

        eb = EdgeBuilder()
        medium_connections = [(0, 3), (3, 6)]
        weak_connections = [(2, 4), (5, 7)]
        cycles = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        edge_table = []
        edge_table.append([100, eb.new_edges_cycles(cycles)])
        edge_table.append([10, medium_connections])
        edge_table.append([1, weak_connections])
        return node_table, edge_table

    
