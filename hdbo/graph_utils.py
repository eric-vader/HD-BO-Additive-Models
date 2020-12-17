import networkx as nx
import copy

from itertools import combinations

# Taken from http://programtalk.com/vs2/?source=python/9061/neuroBN/neuroBN/utils/graph.py
def make_chordal(bn):
    """
    This function creates a chordal graph - i.e. one in which there
    are no cycles with more than three nodes.

    Algorithm from Cano & Moral 1990 ->
    'Heuristic Algorithms for the Triangulation of Graphs'
    """
    chordal_E = list(bn.edges())

    # if moral graph is already chordal, no need to alter it
    if not nx.is_chordal(bn):
        temp_E = copy.copy(chordal_E)
        temp_V = []
        
        temp_G = nx.Graph()
        temp_G.add_edges_from(chordal_E)
        degree_dict = temp_G.degree()
        temp_V = [ v for v, d in sorted(degree_dict, key=lambda x:x[1]) ]
        for v in temp_V:
            #Add links between the pairs nodes adjacent to Node i
            #Add those links to chordal_E and temp_E
            adj_v = set([n for e in temp_E for n in e if v in e and n!=v])
            for a1 in adj_v:
                for a2 in adj_v:
                    if a1!=a2:
                        if [a1,a2] not in chordal_E and [a2,a1] not in chordal_E:
                            chordal_E.append([a1,a2])
                            temp_E.append([a1,a2])
            # remove Node i from temp_V and all its links from temp_E 
            temp_E2 = []
            for edge in temp_E:
                if v not in edge:
                    temp_E2.append(edge)
            temp_E = temp_E2
     
    G = nx.Graph()
    G.add_nodes_from(bn.nodes())
    G.add_edges_from(chordal_E)
    return G

def build_clique_graph(G):
    clique_graph=nx.Graph()
    max_cliques = nx.chordal_graph_cliques(make_chordal(G))
    # The case where there is only 1 max_clique
    if len(max_cliques) == 1:
        clique_graph.add_node(max_cliques.pop())
        return clique_graph
    
    for c1, c2 in combinations(max_cliques, 2):
        intersect = c1.intersection(c2)
        if len(intersect) != 0:
            # we put a minus sign because networkx only allows for MINIMUM Spanning Trees...
            clique_graph.add_edge(c1, c2, weight = -len(intersect))
        else:
            clique_graph.add_node(c1)
            clique_graph.add_node(c2)
    return clique_graph
    
def debug_graph(G):
    pos=nx.spring_layout(G)
    nx.draw(G, pos=pos, cmap = plt.get_cmap('jet'), with_labels=True)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    plt.show()
    plt.clf()
