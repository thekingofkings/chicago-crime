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
    population = D[:,0].reshape(D.shape[0], 1)
    Y = Y / population * 10000
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
    
    
def mvl_fuse_function(models, Y_test):
    pass





import unittest
class MVLTest(unittest.TestCase):
    
    def test_generate_raw_samples(self):
        Y, D, P, T, G = generate_raw_samples()
        assert(Y.max() < 20000) # use crime rate
        assert(Y.shape == (77,1))
        assert(D.shape == (77,8))
        assert(P.shape == (77,10))
        assert(T.shape == (77,8))
        assert(G.shape == (77,8))
        
        
if __name__ == '__main__':
    
    unittest.main()