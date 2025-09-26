import numpy as np
import os
import time
import glob
from numba import jit, njit, float64


class order_params:
    def __init__(self, director_field, velocity_field):
        pass

    def get_vicsek_op(self, velocities, act_strength):
        # obtains the instataneous vicsek order parameter
        order_param = np.sum(velocities)/ (np.shape(velocities)[0] * act_strength )
        return order_param 


    @njit()
    def get_vicsek_director_op(self, directors):
        # input all the n vectors (directors) of the particles and produces the global polar director
        return np.sum(directors) / np.shape(directors)[0]


    #@njit()
    def kdelta(self, x,y):
        if (x==y):
            return 1
        else:
            return 0
        
    #@njit()
    def nvector(self, m):
        n = np.absolute(m)
        mx = max(n)
        pos = 0
        for i in range(len(n)):
            if(mx==n[i]):
                pos = i
        return pos

    #@njit(float64[:](float64[:],float64[:],float64[:]))
    def Qmethod(self, ux,uy,uz):
        
        Q = np.zeros((3,3),dtype=np.float64)
        u = [ux,uy,uz]
        
        for i in range(3):
            for j in range(3):
                for k in range(len(ux)):
                    Q[i][j]  = Q[i][j] + (3/2 * u[i][k] * u[j][k] - self.kdelta(i,j)/2) 

        Q = np.divide(Q , int(len(ux)))
        
        w,v = np.linalg.eigh(Q)
        
        n_vec_pos = self.nvector(w)
        
        n_vect2 = v[int(n_vec_pos)]
        
        #n_vect2_mag = np.sqrt(n_vect2[0]**2 + n_vect2[1]**2 + n_vect2[2]**2)
        
        n_vect2_mag = np.linalg.norm(n_vect2)
        
        n_vect2 = np.divide(n_vect2 , n_vect2_mag) 
        
        return n_vect2
        

    def calOrder(self, n_vt,ux,uy,uz):
        # calculates the magnitude of the Q-tensor order parameter
        S = 0
        for i in range(len(ux)):
            S = S + ( (3/2)*(np.dot(n_vt,np.array([ux[i],uy[i],uz[i]]))**2) -1/2 )
        S = S/int(len(ux))	
        
        return S

