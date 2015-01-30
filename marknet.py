"""

a library of helper functions for working with stochastic matrices with
the NetworkX Python graph theory library

"""
from numpy import *
from scipy import *
from matplotlib.pyplot import *

import networkx as nx


def mat2DiGraph(trans_mat):
	"""
	Given a transition matrix, generate a NetworkX DiGraph object
	corresponding to that transition matrix
	"""

    num_states = trans_mat.shape[0]
    Tg=nx.DiGraph()
    Tg.add_nodes_from(range(num_states))

    # add self-connections
    for node in Tg:
        Tg.add_edge(node, node)

    for ii in range(num_states):
        for jj in range(num_states):
            if trans_mat[ii][jj] > .001:
                Tg.add_edge(ii, jj)
                # Weight by probability of not transitioning
                Tg[ii][jj]['weight'] = (1-trans_mat[ii][jj])
    return Tg

