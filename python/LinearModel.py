# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:01:30 2016

@author: hxw186



linear regression model

The conventional linear model and 
the dynamic coefficient model
"""



from sklearn import linear_model
import matplotlib.pyplot as plt
    


""" =========== standard linear regression model ==================== """


def linearRegression(features, Y):
    """
    learn the linear regression model from features to Y
    output the regression analysis parameters
    plot scatter plot
    """
    

#    mod = sm.OLS(Y, features )
    mod = linear_model.LinearRegression()
    res = mod.fit(features, Y)
    return res
    



""" =========== dynamic coefficients linear regression model ====================

min_W  sum_i ||y - wx||_2^2  + eta sum_ij S_ij ||w_i - w_j||_2^2 + theta ||W||_F^2

"""

import numpy as np
import numpy.random as rnd
import numpy.linalg as la


def dynamicLR(features, Y, S, eta = 1):
    """
    dimensions of each variables
    
        features: N x c
        Y: N x 1
        S: N x N
        W: (N+1) x c
    """
    assert len(Y) == features.shape[0]
    N = len(Y)
    c = features.shape[1]
    W = rnd.rand(N+1, c)
    # append the all 0 w_N
    W[N,] = 0
    S = np.vstack( (S, np.ones(N)) )
    S = np.hstack( (S, np.ones((N+1,1))) )
    
    import sys
    er_prev = sys.maxint
    er = dynamicLR_error(W, features, Y, S, eta)
    
    while (abs(er_prev - er) / er > 0.0001 ):
#        print er
        # update W
        for i in range(N):
            W[i,] = update_wi(features[i,], S[i,], Y[i], W, i, eta)
        er_prev = er
        er = dynamicLR_error(W, features, Y, S, eta)
    
#    print "dynamic LR training finished"
    
#    plt.matshow(W)
    return W




def dynamicLR_error(W, X, Y, S, eta):
    N = len(Y)
    er = la.norm(Y - np.dot( W[0:N,], X.transpose() ) ) ** 2
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if S[i,j] == 1:
                er += eta * la.norm(W[i,] - W[j,]) ** 2
    return er
    
    
def update_wi(xi, Si, yi, W, i, eta):
    """
    update w_i while fix all other w_j in W
    xi: 1 x c
    yi: scalar
    Si: 1 x (N+1)
    W: (N+1) x c
    """
    c = len(xi)
    xih = xi[:,None]    # add one dimension in the end (output is column vector)
    tmp1 = xih * xih.transpose() + eta * np.sum(Si) * np.eye(c)
    tmp2 = yi * xi + np.dot(Si, W)
    assert tmp1.shape == (c, c) and tmp2.shape == (c,)
    return np.dot( la.inv(tmp1), tmp2 )



from foursquarePOI import getFourSquarePOIDistribution, getFourSquareCount
from taxiFlow import getTaxiFlow
from FeatureUtils import *



def prepare_features(features=["poi", "taxi", "demos", "spatiallag"], leaveOneOut=-1):
    
    Y = retrieve_crime_count(year=2013)
    Y = Y.reshape((-1,1))
    if leaveOneOut > 0:
        Y = np.delete(Y, leaveOneOut-1, 0)
    
    if "poi" in features:
        poi_dist = getFourSquareCount(leaveOut=leaveOneOut)
    
    if "taxi" in features:
        F_taxi = getTaxiFlow(leaveOut=leaveOneOut, normalization="bysource")

    if "demos" in features:
        C = generate_corina_features(leaveOut=leaveOneOut)
        demos = ['total population', 'population density', 'disadvantage index', 
                 'residential stability', 'ethnic diversity']
        demos_idx = [C[0].index(ele) for ele in demos]
        D = C[1][:,demos_idx]
        
        popul = C[1][:,0].reshape(C[1].shape[0],1)
        Y = np.divide(Y, popul) * 10000
        
    if "spatiallag" in features:
        W2 = generate_geographical_SpatialLag_ca(leaveOut=leaveOneOut)
        
    f2 = np.dot(W2, Y)
    ftaxi = np.dot(F_taxi, Y)
    
    f = np.ones(f2.shape)
    f = np.concatenate( (f, D, f2, ftaxi, poi_dist), axis=1 )
    
    S = np.zeros(W2.shape)
    S[W2 > 0] = 1
#    S = np.ones(W2.shape)
    
    return Y, f, S
        

if __name__ == '__main__':
    
    
    Y, f, S = prepare_features()
    err1 = []
    err2 = []
    
    for i in range(len(Y)):
        
        Yp, fp, Sp = prepare_features(leaveOneOut=i+1)
        W = dynamicLR(fp, Yp, Sp, 1)
        
        Si = S[i,]
        Si = np.delete(Si, i)
        Si = np.append(Si, 1)
        wi = np.dot(Si, W) / sum(Si)
        
        
        ybar = np.dot(wi, f[i,])
        err1.append( abs(Y[i] - ybar) )
        
        mod = linearRegression(fp, Yp)
        ybar2 = mod.predict(f[i,])
        err2.append( abs(Y[i] - ybar2) )
        
        print i, Y[i,0], ybar, ybar2[0], sum(err1)[0], sum(err2)[0]
        
    print np.mean(err1), np.mean(err2)