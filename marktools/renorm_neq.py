"""
Given row-normalized stochastic matrices, functions to consolidate columns, rows,
and both rows and columns. Unlike the function renorm(), the row consolidation
does not use the steady-state state occupation probabilities but instead weighs
each row in a cluster equally---corresponding to local equilibration over some
timescale proportional to the smallest transition rate between two states that get
consolidated.
"""
from numpy import *
from scipy import *
from matplotlib.pyplot import *

from scipy.linalg import eig

def renorm_col(inmat, k=2, its=1):
    """
    Take a row-normalized transition matrix and 
    k-tuple the columns together 
    NOTE: THE COLUMNS ARE CLUSTERED BY THEIR ORIGINAL SORTING
    
    Parameters (also attributes)
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
    tmat : array
        A renormalized array
    
    """
    
    tmat = copy(inmat)
    for tt in range(its):
        
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
        
    return tmat
            

    
def renorm_row(inmat, k=2, its=1):
    """
    Uniform weighting assumes local equilibration
    Take a row-normalized transition matrix and 
    k-tuple the rows together 
    NOTE: THE ROWS ARE CLUSTERED BY THEIR ORIGINAL SORTING
    
    Parameters (also attributes)
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
    tmat : array
        A renormalized array
    
    """
    
    tmat = copy(inmat).T
    for tt in range(its):
        
        sz = tmat.shape

        num_col = sz[1]

        nmat = zeros( [sz[0], ceil(sz[1]/double(k))] )
        nsz = nmat.shape

        for ii in range(nsz[1]-1):
            nmat[:, ii] = (1./k)*sum(tmat[:,(k*ii):(k*(ii+1))], axis=1)

        if (mod(num_col, k) == 0):
            nmat[:, -1] = (1./k)*sum(tmat[:,-k:], axis=1)
        else:
            nmat[:, -1] = (1./(mod(num_col, k)))*sum(tmat[:,-(mod(num_col, k)):], axis=1)
         
        tmat = nmat
        
    return tmat.T

def renorm_neq(inmat, k=2, its=1):
    """
    Compute a square renormalization on both the rows and 
    columns of a square array. It does not use the equilibirum distribution to
    weight the combinations of rows, but rather uses the assumption of local equilibration
    within each consolidated state (so each row gets equal weighting)
    """
    return renorm_row(renorm_col(inmat, k=k, its=its), k=k, its=its)
