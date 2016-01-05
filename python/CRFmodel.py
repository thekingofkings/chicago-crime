# -*- coding: utf-8 -*-
"""
CRF model

Format the input matrix for the conditional random field (CRF) model. Later, 
these inputs are fed to Matlab for the optimization.


In this file the CRFv1 is also implemeneted, which uses the numpy package to solve 
the following problem.

    min_{alpha, w} ||X alpha - y||_1 + ||F w - y_p||_1


An utility function is separately implemented, which is the one-norm-error solver
for problem with the following form:

    min_{alpha} || X * alpha - Y ||_1

Created on Tue Dec 29 16:05:19 2015

@author: hxw186
"""


from FeatureUtils import *
import numpy as np
from sklearn.preprocessing import scale



""" ==========================================================================
CRF model version 1
    min_{alpha, w} ||X alpha - y||_1 + ||F w - y_p||_1
========================================================================== """



def generateInput_v1(fout=False):
    """
    Generate complete observation matrix
    """
    des, X = generate_corina_features('ca')
    
    F_dist = generate_geographical_SpatialLag_ca()
    F_flow = generate_transition_SocialLag(year=2010, lehd_type=0, region='ca')
    
    
#    X = scale(X)
#    F_dist = scale( F_dist )
#    F_flow = scale( F_flow )

    Y = retrieve_crime_count(year=2010, col=['total'], region='ca')

    Yp = []
    F = []
    n = Y.size
    for i in range(n):
        for j in range(n):
            if i != j:
                Yp.append( Y[i, 0] - Y[j,0] )
                fij = [F_dist[i,j], F_flow[i,j]]
                F.append(fij)
    Yp = np.array(Yp)
    Yp.resize( (Yp.size, 1) )
    F = np.array(F)

    if fout:
        np.savetxt('../matlab/Y.csv', Y, delimiter=',')
        np.savetxt('../matlab/X.csv', X, delimiter=',')
        np.savetxt('../matlab/Yp.csv', Yp, delimiter=',')
        np.savetxt('../matlab/F.csv', F, delimiter=',')

    return X, Y, F, Yp



def leaveOneOut_Input_v1( leaveOut ):
    """
    Generate observation matrix and vectors
    X
    y
    F
    Yp

    Those observations are trimed for the leave-one-out evaluation. Therefore, the leaveOut 
    indicates the CA id to be left out, ranging from 1-77
    """
    
    # get complete X (demographics), and leave one out
    des, X = generate_corina_features('ca')
    X = np.delete(X, leaveOut-1, 0)
    
    # get complete spatial lag, and leave-one-out
    F_dist = generate_geographical_SpatialLag_ca( leaveOut=leaveOut )

    # get complete social lag, and leave one out
    F_flow = generate_transition_SocialLag(year=2010, lehd_type=0, region='ca', leaveOut=leaveOut) 
    
    
#    X = scale(X)
#    F_dist = scale(F_dist)
#    F_flow = scale(F_flow)

    # get complete Y (crime rate), and leave-one-out
    Y = retrieve_crime_count(year=2010, col=['total'], region='ca')
    Y = np.delete(Y, leaveOut-1, 0)

    
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
    
    return X, Y, F, Yp
    





def CRFv1(X, Y, F, Yp):
    """
    version 1 of the CRF-based crime rate model

        min_{alpha, w} ||X alpha - y||_1 + ||F w - y_p||_1
    """
    
    alpha = OneNormErrorSolver(X, Y)
    # print alpha

    w = OneNormErrorSolver(F, Yp)
    # print w

    return alpha, w






def inference_Yi_crfv1( alpha, w, X, Y, F, leaveOut ):
    """
    Make the inference on the value of y_i, 
        
        P(y_i| x_i, F, Y, alpha, w)
    """
    xi = X[leaveOut-1]
    seq = [ np.dot(np.transpose(alpha), xi)[0] ]

    n = Y.size
    i = leaveOut - 1
    for j in range(n):
        if i != j:
            seq.append( (Y[j] + np.dot( np.transpose(w), F[i] ))[0] )

    minv = multAbsTermSolver( seq )
    return minv, np.mean(seq)

    


def CRFv1_leaveOneOut_evaluation():
    """
    Evaluate the CRF v1 with leave-one-out at CA level
    """
    print 'CRF v1 -- potential function is defined on both  1 clique and 2 clique'

    cX, cY, cF, cYp = generateInput_v1()

    error1 = []
    error2 = []
    for leaveOut in range(1, 78):
        X, Y, F, Yp = leaveOneOut_Input_v1( leaveOut )
        alpha, w = CRFv1(X, Y, F, Yp)

        res = inference_Yi_crfv1( alpha, w, cX, cY, cF, leaveOut ) 
        error1.append( abs(res[0] - cY[leaveOut-1][0]) )        
        error2.append( abs(res[1] - cY[leaveOut-1][0]) )        
        print res[0], cY[leaveOut-1][0]

    mae1 = np.mean(error1)
    var1 = np.sqrt( np.var(error1) )
    mre1 = mae1 / Y.mean()

    mae2 = np.mean(error2)
    var2 = np.sqrt( np.var(error2) )
    mre2 = mae2 / Y.mean()

    print 'Use 1-norm inference mae {0}, var {1}, mre {2}'.format( mae1, var1, mre1 )
    print 'Use 2-norm inference mae {0}, var {1}, mre {2}'.format( mae2, var2, mre2 )





""" ==========================================================================
CRF model version 2
    min_{alpha, w} \sum_i^n \sum_j^n 
        | y_i - alpha * x_i - beta * y_j - gamma * f_ij |
========================================================================== """




def generateInput_v2(fout=False):
    """
    Generate complete observation matrix
    """
    des, X = generate_corina_features('ca')
    X = scale(X)
    F_dist = generate_geographical_SpatialLag_ca()
    F_dist = scale(F_dist)
    F_flow = generate_transition_SocialLag(year=2010, lehd_type=0, region='ca')
    F_flow = scale(F_flow)

    Y = retrieve_crime_count(year=2010, col=['total'], region='ca')

    F = []
    n = Y.size
    for i in range(n):
        for j in range(n):
            if i != j:
                fij = np.concatenate( (X[i], np.array( [Y[i][0], F_dist[i,j], F_flow[i,j]]) ), 1)
                F.append(fij)
    F = np.array(F)

    if fout:
        np.savetxt('../matlab/F.csv', F, delimiter=',')

    return Y, F




def leaveOneOut_Input_v2( leaveOut ):
    """
    Generate observation matrix and vectors
    Y, F

    Those observations are trimed for the leave-one-out evaluation. Therefore, the leaveOut 
    indicates the CA id to be left out, ranging from 1-77
    """
    des, X = generate_corina_features('ca')
    X = np.delete(X, leaveOut-1, 0)
    X = scale(X)
    
    F_dist = generate_geographical_SpatialLag_ca( leaveOut=leaveOut )
    F_dist = scale(F_dist)
    F_flow = generate_transition_SocialLag(year=2010, lehd_type=0, region='ca', leaveOut=leaveOut)
    F_flow = scale(F_flow)
    
    Y = retrieve_crime_count(year=2010, col=['total'], region='ca')
    Y = np.delete(Y, leaveOut-1, 0)
    
    F = []
    n = Y.size
    Yd = []
    for i in range(n):
        for j in range(n):
            if i != j:
                fij = np.concatenate( (X[i], np.array( [Y[i,0], F_dist[i,j], F_flow[i,j]] )), 1)
                F.append(fij)
                Yd.append(Y[i])
    F = np.array(F)
    Yd = np.array(Yd)
    Yd.resize( (Yd.size, 1) )
    
    
    return Yd, F





def CRFv2(Yd, F):
    """
    Version 2 of the CRF-based crime predict model
        
        min_{w} || F w - Yd ||_1
    """

    w = OneNormErrorSolver(F, Yd)
    # print w
    
    return w
    
    
    
def inference_Yi_crfv2( w, F, Y, leaveOut ):
    """
    Make the inference on y_i
        
        P(y_i|F_i, Y, F, w)
    """
    
    n = Y.size
    startidx = (leaveOut-1) * (n-1)
    seq = []
    for i in range(n-1):
        for j in range(w.size):
            print (w[j] * F[startidx+i][j])[0], 
        s = np.dot(np.transpose(w), F[startidx + i])[0]
        seq.append( s )
        print s
    
    
    minv = multAbsTermSolver( seq )
    return minv
    
    
    
    
    
def CRFv2_leaveOneOut_evaluation():
    """
    Evaluate the CRF v2
    """
    print 'CRF v2 -- potential function is defined on 2 clique only'
    
    Yc, Fc = generateInput_v2()
    
    error = []
    for leaveOut in range(1, 78):
        Yd, F = leaveOneOut_Input_v2(leaveOut)
        w = CRFv2(Yd, F)
        print w
        
        res = inference_Yi_crfv2(w, Fc, Yc, 1)
        print res, Yc[leaveOut-1, 0]
    
    




    
""" ==========================================================================
Helper function

This section defines commonly used function for basic problem solver.
========================================================================== """

    


def multAbsTermSolver( seq ):
    """
    Solve the following problem.
        
        min_y \sum_i^n |y - a_i|,
    where seq = {a_1, a_2, ..., a_n}
    """
    
    demo = seq[0]
    seq.sort()
    
    import sys
    minval = sys.maxint

    n = len(seq)
    rho = np.ones((n,)) 
    
    # add weight to some terms    
    demoidx = seq.index(demo)
    rho = rho * 7.0 / (n-1)
    rho[demoidx] = 1
    
    vals = []
    # minimum value of the first segment
    vals.append( sum ( [ rho[i] * (a - min(seq)) for i, a in enumerate(seq) ] ) )
    for i in range(1, n-1):
        # piece-wise line kx+b
        k = 0
        b = 0
        for j, a in enumerate(seq):
            if a > seq[i]:
                b += a * rho[j]
                k -= rho[j]
            else:
                b -= a * rho[j]
                k += rho[j]
        if k > 0:
            vals.append( k * seq[i] + b )
        else:
            vals.append( k * seq[i+1] + b )
    
    # mimimum value of the last segment
    vals.append( sum ( [ rho[i] * (max(seq) - a) for i, a in enumerate(seq) ] ) )

    return seq[vals.index( min(vals) )]






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
        obj_prev = sum(abs(np.dot(X, alpha) - Y))
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
        
        obj = sum(abs(np.dot(X, alpha) - Y))

        cnt += 1
        if sum(abs(z - np.dot(X, alpha) + Y)) <= 0.01 and abs(obj_prev - obj) / obj_prev < 0.001:
            break

#    print 'Finished in {0} iterations.'.format( cnt )
    return alpha






if __name__ == '__main__':

    CRFv1_leaveOneOut_evaluation()
#    CRFv2_leaveOneOut_evaluation()

