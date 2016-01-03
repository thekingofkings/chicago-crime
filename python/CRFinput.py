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


def generateInput():
    """
    Generate observation matrix and vectors
    X
    y
    F
    Yp
    """
    
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

    return X, Y, F, Yp
    


def CRFv1(X, Y, F, Yp):
    """
    version 1 of the CRF-based crime rate model

    min_{alpha, w} ||X alpha - y||_1 + ||F w - y_p||_1
    """
    
    alpha = OneNormErrorSolver(X, Y)
    print alpha

    w = OneNormErrorSolver(F, Yp)
    print w


def OneNormErrorSolver(X, Y):
    """
    A solver use ADMM to solve the following optimization problem
    min_{alpha} || X * alpha - Y ||_1
    """
    alpha = np.ones( (X.shape[1], 1) )
    z = np.dot( X, alpha) - Y
    rho = 1
    theta = np.ones( z.shape )

    Xt = X.transpose()
    s, residuals, rank, sv = np.linalg.lstsq( np.dot(Xt, X), Xt )

    cnt = 0
    while True:
        # update alpha
        alpha = np.dot (s, (z + Y + theta))

        
        # update z
        u = np.dot(X, alpha) - Y - theta
        lamb = 1 / rho
        x = np.zeros( z.shape )
        for i in range( u.size ):
            if u[i] >= lamb:
                x[i] = u[i] - lamb
            if u[i] <= - lamb:
                x[i] = u[i] + lamb
            if abs(u[i]) < lamb:
                x[i] = 0

        z = x

        # update theta
        theta += z - np.dot(X, alpha) + Y

        cnt += 1
        if sum(abs(z - np.dot(X, alpha) + Y)) <= 0.1:
            break

    print 'Finished in {0} iterations.'.format( cnt )
    return alpha



if __name__ == '__main__':

    X, Y, F, Yp = generateInput()
    CRFv1(X, Y, F, Yp)
