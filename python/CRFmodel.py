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


def generateInput(fout=False):
    """
    Generate complete observation matrix
    """
    des, X = generate_corina_features('ca')
    F_dist = generate_geographical_SpatialLag_ca()
    F_flow = generate_transition_SocialLag(year=2010, lehd_type=0, region='ca')

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



def leaveOneOut_Input( leaveOut ):
    """
    Generate observation matrix and vectors
    X
    y
    F
    Yp

    Those observations are trimed for the leave-one-out evaluation. Therefore, the leaveOut 
    indicates the CA id to be left out, ranging from 0-76
    """
    
    # get complete X (demographics), and leave one out
    des, X = generate_corina_features('ca')
    X = np.delete(X, leaveOut, 0)   
    
    # get complete spatial lag, and leave-one-out
    F_dist = generate_geographical_SpatialLag_ca( leaveOut=leaveOut )

    # get complete social lag
    countDict, ordkey = generate_transition_SocialLag(year=2010, lehd_type=0, region='ca', rawCount=True) 
    # leave one out in the social lag
    ordkey.remove(leaveOut)
    F_flow = np.zeros( (len(ordkey), len(ordkey)) )
    for srcid in ordkey:
        if srcid in countDict:
            sdict = countDict[srcid]
            if leaveOut in sdict:
                del sdict[leaveOut]
            total = (float) (sum( sdict.values() ))
            for dstid, val in sdict.items():
                if srcid != dstid:
                    if total == 0:
                        F_flow[ordkey.index(srcid)][ordkey.index(dstid)] = 0
                    else:
                        F_flow[ordkey.index(srcid)][ordkey.index(dstid)] = val/total
        else:
            F_flow[ordkey.index(srcid)] = np.zeros( (1, len(ordkey)) )


    # get complete Y (crime rate), and leave-one-out
    Y = retrieve_crime_count(year=2010, col=['total'], region='ca')
    Y = np.delete(Y, leaveOut, 0)

    
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
    print alpha

    w = OneNormErrorSolver(F, Yp)
    print w

    return alpha, w






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





def inference_Yi( alpha, w, X, Y, F, leaveOut ):
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


    minidx = multAbsTermSolver( seq )
    return seq[minidx]

    


def multAbsTermSolver( seq ):
    """
    Solve the following problem.
        
        min_y \sum_i^n |y - a_i|,
    where seq = {a_1, a_2, ..., a_n}
    """
    seq.sort()
    import sys
    minval = sys.maxint

    n = len(seq)
    vals = []
    # minimum value of the first segment
    vals.append( sum ( [ a-min(seq) for a in seq ] ) )
    # mimimum value of the last segment
    vals.append( sum ( [ max(seq)-a for a in seq ] ) )
    for i in range(1, n):
        k = i - (n-i)
        b = 0
        for a in seq:
            if a > seq[i]:
                b += a
            else:
                b -= a
        if k > 0:
            vals.append( k * seq[i] + b )
        else:
            vals.append( k * seq[i+1] + b )

    return vals.index( min(vals) )
    


if __name__ == '__main__':

    cX, cY, cF, cYp = generateInput()


    leaveOut = 1
    X, Y, F, Yp = leaveOneOut_Input( leaveOut )
    alpha, w = CRFv1(X, Y, F, Yp)

    res = inference_Yi( alpha, w, cX, cY, cF, leaveOut ) 
    print res, cY[leaveOut][0]


