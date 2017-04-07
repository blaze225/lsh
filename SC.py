'''
Created on 04-Apr-2017

@author: ethan
'''
import numpy as np
from scipy.interpolate import Rbf,InterpolatedUnivariateSpline,interp1d
import math
# import cv
import sys
import cv2

np.seterr(all='ignore')
  
def euclid_distance(p1,p2):
    return math.sqrt( ( p2[0] - p1[0] ) ** 2 + ( p2[1] - p1[1] ) ** 2 )
    
    
def get_angle(p1,p2):
    """Return angle in radians"""
    return math.atan2((p2[1] - p1[1]),(p2[0] - p1[0]))
    
    
class SC(object):

    HUNGURIAN = 1

    def __init__(self,nbins_r=5,nbins_theta=12,r_inner=0.1250,r_outer=2.0):
        self.nbins_r        = nbins_r
        self.nbins_theta    = nbins_theta
        self.r_inner        = r_inner
        self.r_outer        = r_outer
        self.nbins          = nbins_theta*nbins_r

    def _dist2(self, x, c):
        result = np.zeros((len(x), len(c)))
        for i in xrange(len(x)):
            for j in xrange(len(c)):
                result[i,j] = euclid_distance(x[i],c[j])
        return result
        
    def _get_angles(self, x):
        result = np.zeros((len(x), len(x)))
        for i in xrange(len(x)):
            for j in xrange(len(x)):
                result[i,j] = get_angle(x[i],x[j])
        return result
        
    def compute(self,points,r=None):
        # Get normalized Euclidean distance between each pair of points
        r_array = self._dist2(points,points)
        mean_dist = r_array.mean()
        r_array_n = r_array / mean_dist
        # Create log distance scale #
        r_bin_edges = np.logspace(np.log10(self.r_inner),np.log10(self.r_outer),self.nbins_r)
        print r_bin_edges
        # Create distance histogram #
        r_array_q = np.zeros((len(points),len(points)), dtype=int)
        for m in xrange(self.nbins_r):
            r_array_q += (r_array_n < r_bin_edges[m])
        fz = r_array_q > 0
        # Get angle between each pair of points
        theta_array = self._get_angles(points)
        # Adding 2Pi to negative entries         
        theta_array_2 = theta_array + 2*math.pi * (theta_array < 0)
        theta_array_q = 1 + np.floor(theta_array_2 /(2 * math.pi / self.nbins_theta))
        # Combining distance array and angle array
        BH = np.zeros((len(points),self.nbins))
        for i in xrange(len(points)):
            sn = np.zeros((self.nbins_r, self.nbins_theta))
            for j in xrange(len(points)):
                if (fz[i, j]):                  # If entry in distance histogram is >0
                    abc=int(r_array_q[i, j])
                    xyz=int(theta_array_q[i, j])
                    sn[abc - 1, xyz - 1] += 1
            BH[i] = sn.reshape(self.nbins)   
        return BH

    
