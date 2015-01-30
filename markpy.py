"""
A library of functions for working with stochastic matrices. The convention used is row-normalization, in which the row index (usually "i") indexed the current state, and the column index ("j") indexes the next state
"""
from numpy import *
from scipy import *
from matplotlib.pyplot import *


def getss(tmat):
	"""
	Given a stochastic transition matrix, find the steady state
	and give a warning if it does not exist. Normalize the steady-state
	before returning it.
	"""
    eigset = eig(zmat,left=True,right=False)
    if abs(eigset[0][0]-1.0)>1e-10:
        print("Warning: No eigenvalue equal to one detected. Check transition matrix")
    ss = (eigset[1][0])
    ss = ss/sum(ss)
    return ss