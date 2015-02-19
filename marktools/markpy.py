"""
A library of functions for manipulating stochastic matrices. 
The convention used is row-normalization, in which the row index 
(usually "i") indexed the current state, and the column index ("j") 
indexes the next state.
"""
from numpy import *
from scipy import *
from matplotlib.pyplot import *

from scipy.linalg import eig
from numpy.random import random_sample
from warnings import warn, simplefilter
import networkx as nx

# from marknet import mat2DiGraph
# from renorm_neq import *
from .marknet import *
from .renorm_neq import *


warnings.simplefilter('always', UserWarning)



def discrete_dist(vals, weights, nn=1):
    """
    nn is the number of random values to generate
    """
    bins = cumsum(weights)
    return vals[digitize(random_sample(nn), bins)]

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
    # eigset = eig(tmat, left=True, right=False)
    eigset = eig(tmat.T)

    # check for real part nearly equal to one and 
    # imaginary part nearly equal to zero
    windex = (abs(real(eigset[0]-1.0)) < 1e-14) & (imag(eigset[0]) < 1e-14)
    if any( windex ):
        wind = list(windex).index(True)
    else:
        warn("No eigenvalue equal to one detected. Check transition matrix")
        print(eigset[0])
        wind=0
    ss = (eigset[1].T[wind])
    ss = ss/sum(ss)
    return ss

def lefteig(tmat):
    """
    Given a stochastic transition matrix, find the orthonormal eigenbasis
    if it exists. Eigenvectors and eigenvalues are sorted by the magnitude
    of the eigenvalue

        Parameters
    ----------
    tmat : array
        A stochastic matrix
    """
 
    eigset = eig(tmat.T)
    
    
    sort_ind = argsort(eigset[0])[::-1]
    eig_vals = eigset[0][sort_ind]
    eig_vecs = eigset[1].T[sort_ind,:]
    
    eig_vecs = rownorm(eig_vecs)
    
    return [eig_vals, eig_vecs]


def markov_plot(tmat,scale=1000):
    """ 
    Take a stochastic matrix as input and make a weighted 
    plot. The size of each node is proprotional to its 
    population at steady state

    tmat : array
        A stochastic matrix

    scale : int
        The amount to scale the node size
    """
    ss = getss(tmat)
    sz_array = ss
    
    Tg = mat2DiGraph(tmat)

    pos=nx.circular_layout(Tg)
    nx.draw(Tg, pos, arrows=True,node_size=scale*sz_array)


def matprint(M):
    """
    Makes a pretty printing of a square matrix
    """
    for item in M:
        for val in item:
            print('{0:.3f}'.format(val).rjust(6),end=" ")
        print('\n')

def unhollow(tmat):
    """
    given a nearly stochastic matrix tmat, calculate the 
    self-transition probabilities by summing the off-diagonal 
    entries for each row and subtracting it from one, then 
    putting this number at the diagonal index
    
    tmat : array
        A matrix with zero or otherwise meaningless diagonal elements
    
    umat : array
        The same off diagonal elements as tmat, but with the diagonal
        elements shifted so that the rows all sum to one

    """   
    umat = copy(tmat)
    for (ind,row) in enumerate(umat):
         umat[ind,ind] = 1.0-(sum(row)-row[ind])
    
    return umat

def rownorm(tmat):
    """
    Normalize the rows of a stochastic transition matrix
    using rescaling
    """
    for (ind,row) in enumerate(tmat):
        tmat[ind,:] = row/sum(row)
    return tmat

def rownorm2(inmat):
    """
    Normalize the rows of a stochastic transition matrix
    by re-weighing self-transition probability (diagonal elements)
    """
    tmat = copy(inmat)
    
    for (ind, row) in enumerate(inmat):
        row2 = copy(row)
        row2[ind] = 0.0
        tot = sum(row2)
        
        if tot >= 1.0:
            warn("Index " + str(ind) + ": Negative value on main diagonal. Consider re-weighing transition states")
        
        tmat[ind,ind] = 1.0 - tot
        
    return tmat

def norm_adj(adj):
    """
    Given an unnormalized graph adjacency matrix (with multiple allowed 
    paths between a pair of nodes), normalize the rows correctly:
    
    1. All zero rows are replaced with their transpose
    
    2. All zero rows and all zero columns are replaced with unit
        entries on the diagonal (these states are basically subgraphs)
    
    Parameters
    ----------
    adj : array
        An adjacency matrix for a graph
    

    Returns
    -------
    nadj : array
        a row-normalized stochastic matrix generated from adj

    """
    
    nadj = copy(adj)
    for ii in range(2):
        row_sum = (sum(double(adj),axis=1))
        bad_locs = where(row_sum==0.0)
        # fix zero rows via transposition (remove directedness)
        for loc in bad_locs:
            nadj[loc,:] = copy(nadj[:,loc].T)
        nadj = nadj.T
    nadj = nadj.T
    
    # replace empty states with unit transitions
    for (ind, row) in enumerate(nadj):
        if sum(row)<1e-14:
            nadj[ind,ind] = 1.0    
    nadj = rownorm(nadj)
    
    return nadj

def swapper(inmat, curr, targ):
    """
    Rearranges states in a normalized transition matrix. Returns
    a copy of the array
    
    tmat : array
        The transition matrix to be modified
    curr : int
        index of state to be swapped
    targ : int
        index of desired location at which curr will be inserted
    """
    tmat = copy(inmat)
    
    (old_row, old_column) = (copy(tmat[:,targ]), copy(tmat[targ,:]))
    (tmat[:,targ], tmat[targ,:]) = (copy(tmat[:,curr]), copy(tmat[curr,:]))
    (tmat[:,curr], tmat[curr,:]) = (old_row, old_column)
    
    return tmat

def make_bethe(Z, k, stay_rate=1.0):
    """
    Generate the undirected adjacency matrix of a Bethe lattice
    for a given coordination number and finite shell number
    
    Z : int
        The number of links out of every node
        
    k : int
        The depth of the tree, or the number of "shells"
        
    stay_rate : double
        A weight for self-transitions in the graph. Default they have
        equal probability to transitions out of state. This value is zero 
        in many standard treatments of Cayley maps
        
    adjmat : array
        An adjacency matrix for the lattice
    
    """
    
    N = 1+Z*((Z-1.)**k - 1.)/(Z-2.)
    num_leaves = Z*(Z-1)**(k-1)

    adjmat = zeros([N,N])

    # populate upper triangle first
    for (ind, row) in enumerate(adjmat):
        if (ind != 0):
            adjmat[ind, ((Z-1)*(ind+1)-(Z-3)):((Z-1)*ind+2*(Z-1)-(Z-3))] = 1.0
    adjmat[0,1:(Z+1)] = 1.0
    
    adjmat = adjmat + adjmat.T
    
    # seed with input self-transition rate
    for (ind, row) in enumerate(adjmat):
        adjmat[ind,ind] = stay_rate

    # make leaves metastable to preserve detailed balance
    adjmat[-num_leaves:,:] = adjmat[:,-num_leaves:].T
    
    for (ind,row) in enumerate(adjmat):
        if (ind >= (N-num_leaves)):
            adjmat[ind,ind] += (Z-1.0)

    adjmat = rownorm(adjmat)
    
    return adjmat

def merge_states(inmat, s_inds):
    """
    Merge the tuple of states given by s_inds for a
    stochastic transition matrix (row-normalized)
    
    inmat: array
        A row-normalized transition matrix
    
    s_inds : tuple
        The set of states (at least two) that will be 
        merged. 

    """
    
    tmat = copy(inmat)
    targ_ind = min(s_inds)
    
    ss = getss(tmat)
    weights = array([ss[index] for index in s_inds])
    tot_weight = sum(weights)
    
    row = tmat[s_inds, :]
    
    # correctly merge the rows
    new_row = ( array(weights).dot(tmat[s_inds,:]) )/tot_weight
    new_row = real(new_row)
    
    tmat = delete(tmat, s_inds, axis=0)
    tmat = insert(tmat, targ_ind, new_row, axis=0)
    
    # do the columns
    
    cols = tmat[:, s_inds]
    new_col = sum(cols, axis=1)
    tmat = delete(tmat, s_inds, axis=1)
    tmat = insert(tmat, targ_ind, new_col, axis=1)

    return tmat

def renorm(inmat, k=2, its=1):
    """
    Take a row-normalized transition matrix and rescale the
    probabilities by reducing the dimensionality by a factor of k
    
    Notes: the columns are clustered by whatever sorting they have to begin with
           the weights of row pairings are determined using the equilibrium probabilities of each node
    
    FIXES: This function should be simplified to just call merge_states, the
            part is figuring out odd-shaped matrices. 

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
        
        # get the steady-state
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


