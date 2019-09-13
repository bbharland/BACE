
class EdgeBuilder(object):
    """
    Provides methods to construct edge lists from node lists
        
    Edges are added, "first come first served".  Subsequent attempts
    to add the same edge are ignored

    Methods:
        Return:          Method name:
        -------          ------------
        (ordered tuple)  new_edge(node1, node2)
        (tuple list)     new_edges_pairs(pair_list)
        (tuple list)     new_edges_series(nodes)
        (tuple list)     new_edges_cycle(nodes)
        (tuple list)     new_edges_cycles(cycle_list)
        (tuple list)     new_edges_whiskers(central_node, whisker_list)
        (tuple list)     new_edges_fully_connected(nodes)
        ()               draw_graph()
    """
    # existing_edges = None
    
    def __init__(self):
        self.existing_edges = []

    def new_edge(self, node1, node2):
        "Return single edge, sorted by node"
        edge = tuple(sorted([node1, node2]))
        if edge not in self.existing_edges:
            # self.existing_edges.extend([edge])
            self.existing_edges.append(edge)
            return edge
        else:
            return None

    def new_edges_pairs(self, pair_list):
        "Return list of edges, linking up list of pairs"
        new_edges = []
        for pair in pair_list:
            edge = self.new_edge(*pair)
            if edge is not None:
                new_edges.append(edge)
        return new_edges

    def new_edges_series(self, nodes):
        "Return list of edges, linking up nodes in series"
        new_edges = []
        node = nodes[0]
        for next_node in nodes[1:]:
            edge = self.new_edge(node, next_node)
            if edge is not None:
                new_edges.append(edge)
            node = next_node
        return new_edges

    def new_edges_cycle(self, nodes):
        "Return a list of edges that connect node_list in cycle"        
        new_edges = self.new_edges_series(nodes)
        cycle_edge = self.new_edge(nodes[0], nodes[-1])
        if cycle_edge is not None:
            new_edges.append(cycle_edge)
        return new_edges

    def new_edges_cycles(self, cycle_list):
        "Do more than one cycle at a time"
        new_edges = []
        for cycle in cycle_list:
            cycle_edges = self.new_edges_cycle(cycle)
            new_edges.extend(cycle_edges)
        return new_edges

    def new_edges_whiskers(self, central_node, whisker_nodes):
        "Return a list of edges joining central node to each whisker"
        new_edges = []
        for whisker in whisker_nodes:
            edge = self.new_edge(central_node, whisker)
            if edge is not None:
                new_edges.append(edge)
        return new_edges

    def new_edges_fully_connected(self, nodes):
        "Return a list of edges that connect every pair of nodes"
        new_edges = []
        for i in nodes:
            for j in nodes:
                if j > i:
                    edge = self.new_edge(i, j)
                    if edge is not None:
                        new_edges.append(edge)
        return new_edges

    def draw_graph(self):
        "Draw graph of current existing_edges"
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.Graph()
        nodes = sorted(set([n for e in self.existing_edges for n in e]))
        node_list = [str(n) for n in nodes]
        labels = {node: node for node in node_list}
        for node in node_list: 
            G.add_node(node)
    
        for i, j in self.existing_edges:
            G.add_edge(str(i), str(j))

        pos = nx.spring_layout(G)
        _ = nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                   node_size=1000, alpha=1)
        _ = nx.draw_networkx_labels(G, pos, labels, font_size=12, alpha=1)
        _ = nx.draw_networkx_edges(G, pos, edgelist=G.edges, alpha=0.5, width=3)
        plt.axis('off')
        plt.show()
