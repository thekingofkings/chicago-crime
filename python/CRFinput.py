# -*- coding: utf-8 -*-
"""
CRFinput

Format the input matrix for the conditional random field (CRF) model. Later, 
these inputs are fed to Matlab for the optimization.


Created on Tue Dec 29 16:05:19 2015

@author: hxw186
"""


from FeatureUtils import *
import numpy as np



if __name__ == '__main__':
    
    des, X = generate_corina_features('ca')
    
    F_dist = generate_geographical_SpatialLag_ca()
    F_flow = generate_transition_SocialLag(year=2010, lehd_type=0, region='ca')
        
    Y = retrieve_crime_count(year=2010, col=['total'], region='ca')

    np.savetxt('../matlab/Y.csv', Y, delimiter=',')
    np.savetxt('../matlab/X.csv', X, delimiter=',')
    
    n = Y.size
    Yp = []
    F = []
    for i in range(n):
        for j in range( n):
            if i != j:
                Yp.append( Y[i,0] - Y[j,0] )
                fij = [F_dist[i,j], F_flow[i,j]]
                F.append(fij)
    Yp = np.array(Yp)
    Yp.resize( (Yp.size, 1) )
    F = np.array(F)
    
    np.savetxt('../matlab/Yp.csv', Yp, delimiter=',')
    np.savetxt('../matlab/F.csv', F, delimiter=',')
    