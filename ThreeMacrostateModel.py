import numpy as np
from BACEModel import BACEModel
from EdgeBuilder import EdgeBuilder

class ThreeMacrostateModel(BACEModel):
    """
    A model with 12 microstates grouped into three obvious macrostates

    * C matrix values are specified

    * this model uses EdgeBuilder to construct model *
         -> node_table = [[node, self_transitions], ...]
         -> edge_table = [[transition_count, edge_list], ...]
    """
    def __init__(self, figdir='./'):
        "This calls self._init_C_matrix()"
        super().__init__(figdir=figdir)
        
    def calc_log_BF_matrix(self):
        super().calc_log_BF_matrix()
        
    def merge_best_pair_states(self):
        super().merge_best_pair_states()

    def draw_graph_cbf(self, title='tmp.png'):
        super().draw_graph_cbf(edge_factor=0.1, title=title)

    def draw_graph_counts(self, title='tmp.png'):
        super().draw_graph(edge_factor=0.1, edge_matrix=self.C, title=title)

    def draw_graph_counts_ax(self, ax):
        super().draw_graph_ax(ax, edge_factor=0.1, edge_matrix=self.C)

    def _init_C_matrix(self):
        """
        Choose to use node_table, edge_table to fill C matrix
        """
        node_table, edge_table = self._build_node_edge_table()
        super()._init_C_from_node_edge_table(node_table, edge_table)
        
    def _build_node_edge_table(self):
        """
        Model specified here!
        
        Return: 
            node_table : [[node, self_transitions], ...]
            edge_table : [[transition_count, [edge_list]], ...]        
        """
        # parameters that define this model
        self_transitions = 500
        count_cycle_edges = 100
        count_cross_edges = 50
        count_central_edges = 5

        # node_table:
        node_table = [[node, self_transitions] for node in range(12)]
        
        # edge_table:
        eb = EdgeBuilder()

        # the edges in the cycles
        cycle_list = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        cycle_edges = eb.new_edges_cycles(cycle_list)

        # do the criss-crosses in each cycle
        cross_edges = []
        for cycle in cycle_list:
            cross_edges_cycle = eb.new_edges_fully_connected(cycle)
            cross_edges.extend(cross_edges_cycle)

        # fully connect the central nodes
        central_edges = eb.new_edges_fully_connected([0, 1, 4, 5, 8, 9])

        # build the weighted edge table with the counts
        edge_table = []
        edge_table.append([count_cycle_edges, cycle_edges])
        edge_table.append([count_cross_edges, cross_edges])
        edge_table.append([count_central_edges, central_edges])
        return node_table, edge_table
