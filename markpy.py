"""
A library of functions for working with stochastic matrices. 
The convention used is row-normalization, in which the row index 
(usually "i") indexed the current state, and the column index ("j") 
indexes the next state.
"""
from numpy import *
from scipy import *
from matplotlib.pyplot import *

from scipy.linalg import eig
import networkx as nx

from marknet import mat2DiGraph

def getss(tmat,rt=False):
    """
    Given a stochastic transition matrix, find the steady state
    and give a warning if it does not exist. Normalize the steady-state
    before returning it.

        Parameters
    ----------
    tmat : array
        A stochastic matrix

    rt : Boolean
    	Default False, if rt=True the right eigenvectors are used to
    	compute the steady-state

    """
    
    if rt:
        tmat = tmat.T
    else:
        pass
    eigset = eig(tmat, left=True, right=False)

    # check for real part nearly equal to one and 
    # imaginary part nearly equal to zero
    windex = (abs(real(eigset[0]-1.0)) < 1e-14) & (imag(eigset[0]) < 1e-14)
    if any( windex ):
        wind = list(windex).index(True)
    else:
        print("Warning: No eigenvalue equal to one detected. Check transition matrix")
        print(eigset[0])
        wind=0
    ss = (eigset[1][wind])
    ss = ss/sum(ss)
    return ss


def markov_plot(tmat):
    """ 
    Take a stochastic matrix as input and make a weighted 
    plot. The size of each node is proprotional to its 
    population at steady state
    """
    ss = getss(tmat)
    sz_array = ss
    
    Tg = mat2DiGraph(tmat)

    pos=nx.spring_layout(Tg)
    nx.draw(Tg, pos, arrows=True,node_size=1000*sz_array)


def matprint(M):
    """
    Makes a pretty printing of a square matrix
    """
    for item in M:
        for val in item:
            print('{0:.3f}'.format(val).rjust(6),end=" ")
        print('\n')


def renorm(inmat, k=2, its=1):
    """
    Take a row-normalized transition matrix and rescale the
    probabilities by reducing the dimensionality by a factor of k
    
    Notes: the columns are clustered by whatever sorting they have to begin with
           the weights of row pairings are determined using the equilibrium probabilities of each node
    
    Parameters
    ----------
    inmat : array
        An graph adjacency matrix to be renormalized
        
    k : int
        The number of states to cluster in each step
    
    its : int
        The number of times to apply the 
        renormalization operation
    
    
    Returns
    -------
    kmat : array
        A renormalized array
    
    """

    zmat = copy(inmat)
    
    # Join columns together first
    for tt in range(its):
        
        tmat = copy(zmat)
        
        sz = tmat.shape

        num_col = sz[1]

        nmat = zeros( [sz[0], ceil(sz[1]/double(k))] )
        nsz = nmat.shape

        for ii in range(nsz[1]-1):
            nmat[:, ii] = sum(tmat[:,(k*ii):(k*(ii+1))], axis=1)

        if (mod(num_col, k) == 0):
            nmat[:, -1] = sum(tmat[:,-k:], axis=1)
        else:
            nmat[:, -1] = sum(tmat[:,-(mod(num_col, k)):], axis=1)
         
        tmat=nmat
        
        ss = getss(zmat)

        # pair the rows
        kmat = copy(nmat).T

        sz = kmat.shape

        num_col = sz[1]

        qmat = zeros( [sz[0], ceil(sz[1]/double(k))] )
        nsz = qmat.shape

        for ii in range(nsz[1]-1):

            cols = kmat[:,(k*ii):(k*(ii+1))]
            weights = ss.T[(k*ii):(k*(ii+1))]
            weights = weights/sum(weights)
            
            qmat[:, ii] = sum( (ww*item for ww, item in zip( weights, cols.T)) , axis=0).T

        if (mod(num_col, k) == 0):
            
            cols = kmat[:,-k:]
            weights = ss.T[-k:]
            weights = weights/sum(weights)
            
            qmat[:, -1] = sum( (ww*item for ww, item in zip( weights, cols.T)) , axis=1).T
                        
        else:
            
            cols = kmat[:,-(mod(num_col, k)):]
            weights = ss.T[-(mod(num_col, k)):]
            weights = weights/sum(weights)
            
            qmat[:, -1] = sum( (ww*item for ww, item in zip( weights, cols.T)) , axis=1).T
                     
        zmat = qmat.T
        
    
    return zmat


