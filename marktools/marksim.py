"""
A library of functions generating stochastic trajectories and quantifying
their thermodynamics properties. 
The convention used is row-normalization, in which the row index 
(usually "i") indexed the current state, and the column index ("j") 
indexes the next state.
"""
from numpy import *
from scipy import *
from matplotlib.pyplot import *

from scipy.linalg import eig
import networkx as nx

# from renorm_neq import *
from .renorm_neq import *
from .markpy import *
from numpy.random import random_sample

def simtraj(trans_mat, tsteps, stt=0):
    """
    Simulation of a trajectory undergoing transitions governed by a jump
    probability matrix
    
    Parameters
    ----------
    
    trans_mat : array
        An NxN matrix of transition probabilities. Current state is 
        given by rows, next state is given by columns
    tsteps : int
        The number of timesteps that elapse
    stt : int
        The state to in which to start the system
        
    Returns
    ----------
    seq : array
        A sequence of states corresponding to a random walk
    
    """
    seq = zeros(tsteps)
    curr = stt
    
    nstates = trans_mat.shape[0]
    states = array([ii for ii in range(nstates)])
    
    for tt in range(tsteps):
        seq[tt] = curr
        weights = copy(trans_mat[curr, :])
        curr = discrete_dist(states, weights, nn=1)
    return seq

def ss_ent(tmat):
    """
    Calculate the steady-state entropy production of a row-
    normalized stochastic transition matrix
    """
    ss = getss(tmat)
    ent = 0.0
    for (state, prob) in enumerate(ss):

        ratio = (tmat[state,:]/tmat[:,state])

        # account for non-connecting elements
        ratio[isnan(ratio)] = 1.0

        ent += prob*( tmat[state, :].dot( log(ratio) ) )
    return ent

def path_ent(tmat, traj):
    """
    Given a trajectory of states sampled from an MSM, calculates the total entropy
    increase of the system along the path
    
    tmat : array
        A transition matrix
    
    traj : array
        A sequence of indices visited
        
    ent : double
        The total stepwise entropy production rate along the path
    """
    traj = array(traj)
    jumps = zip(traj[:-1],traj[1:])
    jump_list = [item for item in (jumps)]

    ent = list()    
    for jump in jump_list:
        prv = int(jump[0])
        nxt = int(jump[1])
        ent.append(log( tmat[prv, nxt]/tmat[nxt, prv] ))
    return list(ent)
