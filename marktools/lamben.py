from numpy import *
from scipy import *
from matplotlib.pyplot import *

from scipy.linalg import eig
from numpy.linalg import matrix_power
from numpy.linalg import inv

# from markpy import *
from .markpy import *

class LambEn:
    """
    Generate a lambda-ensemble from a stochastic (row-normalized)
    transition matrix and a biasing parameter lambda
    
    
    
    Parameters (also attributes)
    ----------
    
    self.trans_mat : array
        A stochastic matrix with row-normalization
    
    self.lmda : double
        An exponent lambda corresponding to the amount 
        of biasing in hte lambda ensemble
        
    Attributes
    ----------
    
    self.ss : array
        the steady state distribution of the transition matrix
    
    self.tiltmat : array
        a "tilted" transition matrix under the lambda ensemble
    
    """
    
    def __init__(self, tmat, lmda):
        self.trans_mat = tmat
        self.lmda = lmda
        self.tiltmat = self.tilt()

        eigset = eig(self.trans_mat,left=True,right=False)
        if abs(eigset[0][0]-1.0)>1e-10:
            print("Warning: No eigenvalue equal to one detected. Check transition matrix")

        self.ss = eigset[1][0]

        
    def tilt(self):
        """ 
        Generates a normalized "tilted" transition matrix 
        in the lambda ensemble

        Returns
        -------
        tilted : array
            A tilted stochastic matrix

        """

        tilted = (self.trans_mat**self.lmda)*(self.trans_mat.T**(1.0-self.lmda))


        # # This normalization portion needs to be checked since
        # # it doesn't quite seem right for the discrete case
        # qq2 = copy(tilted)
        
        # eigset = lefteig(qq2)
        # max_w = eigset[0][0]
        # ss = eigset[1][0]

        # umat = zeros(qq2.shape)
        # for (ind, row) in enumerate(umat):
        #     umat[ind,ind] = ss[ind]

        # compo = (umat.dot(qq2.dot(inv(umat))))
        # compo = real(compo)
        # compo = unhollow(compo)

        compo = unhollow(tilted)

        return compo
    
    
    def Z_lam(self, tsteps):
        """
        Define the partition function at a given time by 
        propagating a tilted matrix for [tstep] timsteps.
        Note that the tilting operation breaks row 
        normalization, implying that the matrix powers can't
        be used to actuall generate trajectories

        Parameters
        ----------

        tsteps : integer
            The integer number of timesteps that elapse so that partition
            function may be calculated

        Returns
        -------
        zz : double
            The numerical value of the partition function
        """

        oo = ones(len(self.ss))

        prop_mat = matrix_power(self.tiltmat, tsteps)

        zz = self.ss.dot(prop_mat.dot(oo))
        return zz

    
    def fen(self, tsteps):
        """ Compute the free energy at a given time """
        Z = self.Z_lam(tsteps)
        return -(1./tsteps)*log(Z)
    
    
    def px(self, tsteps, time, state_index):
        """
        Calculate the probability of being in state [state_index]
        at time [time] under the ensemble. Note: in order to not violate causality,
        time < tsteps/2

        Parameters
        ----------
        tsteps : int
            The number of timesteps that elapse

        time : int
            The probability at which the time is calculated

        state_ind : int
            The index of the Markov state of interest

        Returns
        -------
        p_state : double
            The probability of being in the Markov state given by state_index

        """
        
        if (time >= tsteps):
            print ("Warning: time is greater than number of timesteps.")
        
        mm = zeros(self.trans_mat.shape)
        mm[state_index][state_index] = 1.0
        
        oo = ones(len(self.ss))
        
        prop_mat = matrix_power(self.tiltmat, time)
        
        adj_prop_mat = matrix_power(self.tiltmat, tsteps-time)
        
        mid_op = adj_prop_mat.dot(mm.dot(prop_mat))
        
        return (1.0/self.Z_lam(tsteps))*self.ss.dot(mid_op.dot(oo))

# demo code (if module run as script)
if __name__ == "__main__":
    tmat = rownorm(rand(3,3))
    tiltmat = LambEn(tmat,.7).tiltmat
    matprint(tmat)
    print('\n')
    matprint(tiltmat)

        