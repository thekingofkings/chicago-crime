#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:16:43 2016

@author: hxw186

Evaluate multi-view learning framework.
"""

import numpy as np
import statsmodels.api as sm
from sklearn.cross_validation import LeaveOneOut

import sys
sys.path.append("../")

from graph_embedding import get_graph_embedding_features
from feature_evaluation import extract_raw_samples



def generate_raw_samples():
    """
    Generate raw features for all samples.
    
    Returns
    -------
    Y : Numpy.Array
        Crime counts
    D : Numpy.Array
        Demo features
    P : Numpy.Array
        POI features
    T : Numpy.Array
        Taxi flow graph embedding
    G : Numpy.Array
        Geographic graph embedding
    """   
    Y, D, P, Tf, Gd = extract_raw_samples()
    T = get_graph_embedding_features('taxi_all.txt')
    G = get_graph_embedding_features('geo_all.txt')
    return Y, D, P, T, G
    


def NBmodel(train_idx, Y, X):
    nbm = sm.GLM(Y[train_idx], X[train_idx], family=sm.families.NegativeBinomial())
    nb_res = nbm.fit()
    return nb_res
    
    
def taxi_view_model(train_idx, Y, T):
    return NBmodel(train_idx, Y, T)
    
    
def poi_view_model(train_idx, Y, P):
    return NBmodel(train_idx, Y, P)
   
    
def demo_view_model(train_idx, Y, D):
    return NBmodel(train_idx, Y, D)
    
    
def geo_view_model(train_idx, Y, G):
    return NBmodel(train_idx, Y, G)
    
    
def mvl_fuse_function(models, train_idx, Y):
    pass





import unittest
class MVLTest(unittest.TestCase):
    
    def test_generate_raw_samples(self):
        Y, D, P, T, G = generate_raw_samples()
        assert(Y.max() < 20000 and np.mean(Y) > 1000) # use crime rate
        assert(Y.shape == (77,1))
        assert(D.shape == (77,8))
        assert(P.shape == (77,10))
        assert(T.shape == (77,8))
        assert(G.shape == (77,8))
        
        
    def test_view_model_independently(self):
        Y, D, P, T, G = generate_raw_samples()
        loo = LeaveOneOut(len(Y))
        T = sm.add_constant(T, prepend=False)
        P = sm.add_constant(P, prepend=False)
        D = sm.add_constant(D, prepend=False)
        G = sm.add_constant(G, prepend=False)
        ter = []
        per = []
        der = []
        ger = []
        for train_idx, test_idx in loo:
            nbm = taxi_view_model(train_idx, Y, T)
            ybar = nbm.predict(T[test_idx])
            ter.append(np.abs(ybar - Y[test_idx]))
            
            nbm = poi_view_model(train_idx, Y, P)
            ybar = nbm.predict(P[test_idx])
            per.append(np.abs(ybar - Y[test_idx]))
            
            nbm = demo_view_model(train_idx, Y, D)
            ybar = nbm.predict(D[test_idx])
            der.append(np.abs(ybar - Y[test_idx]))
            
            nbm = demo_view_model(train_idx, Y, G)
            ybar = nbm.predict(G[test_idx])
            ger.append(np.abs(ybar - Y[test_idx]))
            
        tmre = np.mean(ter) / np.mean(Y)
        print "Taxi MRE: {0}".format(tmre)
        assert( tmre < 0.5 )
        
        pmre = np.mean(per) / np.mean(Y)
        print "POI MRE: {0}".format(pmre)
        assert( pmre < 0.5 )
        
        dmre = np.mean(der) / np.mean(Y)
        print "Demo MRE: {0}".format(dmre)
        assert( dmre < 0.32 )
        
        gmre = np.mean(ger) / np.mean(Y)
        print "Geo MRE: {0}".format(gmre)
        assert( gmre < 0.5 )
    
            
        
        
        
    def test_liner_combination_model(self):
        """
        Test a simple linear combination model.
        
        We concatenate the feature vectors from four different views into one
        vector. Then train a NB model on this concatenated vector `X`.
        """
        Y, D, P, T, G = generate_raw_samples()
        X = np.concatenate((D,P,T,G), axis=1)
        assert( X.shape == (77, 34) )
        X = sm.add_constant(X, prepend=False)
        loo = LeaveOneOut(len(Y))
        er = []
        for train_idx, test_idx in loo:
            nbm = NBmodel(train_idx, Y, X)
            ybar = nbm.predict(X[test_idx])
            er.append(np.abs(ybar - Y[test_idx]))
        mre = np.mean(er) / np.mean(Y)
        print "Simple combine model MRE: {0}".format(mre)
        assert( mre > 0.235 )
        
    
        
        
if __name__ == '__main__':
    
    unittest.main()