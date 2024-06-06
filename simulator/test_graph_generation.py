import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# G = nx.gnp_random_graph(10, 0.1, directed=True)

# DAG = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])

# print(nx.is_directed_acyclic_graph(DAG))

# print(nx.is_weakly_connected(DAG))

# print(nx.adjacency_matrix(DAG))


def generate_random_connected_dag(numOfNodes, percentOfEdges):
    ''' Generates a random connected directed acyclic graph containing 'numOfNodes' number of nodes and possibly 'percentOfEdges' percent of total possible edges.

    Procedure:
    - Create a full lower triangular matrix (assured to be DAG)
    - Randomly remove edges if that edge doesn't loose the connectivity and DAG-ness of the graph
    '''
    adj_mat = np.tri(numOfNodes, numOfNodes, -1)

    G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)

    totalNumOfEdges = (numOfNodes * (numOfNodes - 1)) / 2
    edgesThreshold = percentOfEdges * totalNumOfEdges 
    triesThreshold = totalNumOfEdges
    numOfTries = 0

    while G.number_of_edges() > edgesThreshold and numOfTries < triesThreshold:
        edges = list(G.edges)

        random_edge = random.choice(edges)

        G.remove_edge(random_edge[0], random_edge[1])

        if not(nx.is_weakly_connected(G) and nx.is_directed_acyclic_graph(G)):
            G.add_edge(random_edge[0], random_edge[1])

        numOfTries += 1

    # print(nx.to_numpy_array(G))
    # print(nx.is_weakly_connected(G))
    # print(nx.is_directed_acyclic_graph(G))
    # nx.draw(G)
    # plt.savefig('data/graph.png')

    nodes = list(G.nodes)
    edges = list(G.edges)

    return nodes, edges


generate_random_connected_dag(6, 0.3)