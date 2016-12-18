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
from Crime import Tract

np.set_printoptions(suppress=True)


def generate_raw_samples(year=2012):
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
    Y, D, P, Tf, Gd = extract_raw_samples(year)
    T = get_graph_embedding_features('taxi_all.txt')
    G = get_graph_embedding_features('geo_all.txt')
    return Y, D, P, T, G
    


def NBmodel(train_idx, Y, X):
    """
    Train a negative binomial model
    
    Return
    ------
    nb_res : the trained negative binomial model.
    y_bar : a numpy.array, the prediction on training samples
    """
    nbm = sm.GLM(Y[train_idx], X[train_idx], family=sm.families.NegativeBinomial())
    nb_res = nbm.fit()
    return nb_res, nb_res.predict(X[train_idx])
    
    
def taxi_view_model(train_idx, Y, T):
    return NBmodel(train_idx, Y, T)
    
    
def poi_view_model(train_idx, Y, P):
    return NBmodel(train_idx, Y, P)
   
    
def demo_view_model(train_idx, Y, D):
    return NBmodel(train_idx, Y, D)
    
    
def geo_view_model(train_idx, Y, G):
    return NBmodel(train_idx, Y, G)
    
    
def mvl_fuse_function(models, train_idx, Y):
    newX = np.ones((len(train_idx), 1))
    for nb_res, y_train in models:
        ytrain = y_train.reshape((len(y_train), 1))
        newX = np.concatenate((newX, ytrain), axis=1)
    lm = sm.GLM(Y[train_idx], newX, family=sm.families.Gaussian())
    lm_res = lm.fit()
    return lm_res
        




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
            nbm, yp = taxi_view_model(train_idx, Y, T)
            ybar = nbm.predict(T[test_idx])
            ter.append(ybar - Y[test_idx])
            
            nbm, yp = poi_view_model(train_idx, Y, P)
            ybar = nbm.predict(P[test_idx])
            per.append(ybar - Y[test_idx])
            
            nbm, yp = demo_view_model(train_idx, Y, D)
            ybar = nbm.predict(D[test_idx])
            der.append(ybar - Y[test_idx])
            
            nbm, yp = demo_view_model(train_idx, Y, G)
            ybar = nbm.predict(G[test_idx])
            ger.append(ybar - Y[test_idx])
            
        tmre = np.mean(np.abs(ter)) / np.mean(Y)
        print "Taxi MRE: {0}".format(tmre)
        assert( tmre < 0.5 )
#        self.visualize_prediction_error(ter, Y, "Taxi view")
        
        pmre = np.mean(np.abs(per)) / np.mean(Y)
        print "POI MRE: {0}".format(pmre)
        assert( pmre < 0.8 )
#        self.visualize_prediction_error(per, Y, "POI view")
        
        dmre = np.mean(np.abs(der)) / np.mean(Y)
        print "Demo MRE: {0}".format(dmre)
        assert( dmre < 0.8 )
#        self.visualize_prediction_error(der, Y, "Demo view")
        
        gmre = np.mean(np.abs(ger)) / np.mean(Y)
        print "Geo MRE: {0}".format(gmre)
        assert( gmre < 0.5 )
#        self.visualize_prediction_error(ger, Y, "Geo view")
    
            
        
        
        
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
            nbm, yp = NBmodel(train_idx, Y, X)
            ybar = nbm.predict(X[test_idx])
            y_error = ybar - Y[test_idx]
#            if np.abs(y_error / Y[test_idx]) > 0.8:
#                print test_idx, ybar, Y[test_idx]
            er.append(y_error)
        mre = np.mean(np.abs(er)) / np.mean(Y)
        print "Simple combine model MRE: {0}".format(mre)
        assert( mre > 0.235 )
#        self.visualize_prediction_error(er, Y, "Concatenate multiple views")
        
    
    
    def test_mvl_fuse_function(self):
        Y, D, P, T, G = generate_raw_samples()
        T = sm.add_constant(T, prepend=False)
        P = sm.add_constant(P, prepend=False)
        D = sm.add_constant(D, prepend=False)
        G = sm.add_constant(G, prepend=False)
        loo = LeaveOneOut(len(Y))
        er = []
        for train_idx, test_idx in loo:
            tm = taxi_view_model(train_idx, Y, T)
            pm = poi_view_model(train_idx, Y, P)
            gm = geo_view_model(train_idx, Y, G)
            dm = demo_view_model(train_idx, Y, D)
            models = [tm, pm, gm, dm]
            lm = mvl_fuse_function(models, train_idx, Y)
            
            
            tm_test = tm[0].predict(T[test_idx])
            pm_test = pm[0].predict(P[test_idx])
            gm_test = gm[0].predict(G[test_idx])
            dm_test = dm[0].predict(D[test_idx])
            
            newX_test = np.array([1, tm_test, pm_test, gm_test, dm_test])
            ybar = lm.predict(newX_test)
            y_error = ybar - Y[test_idx]
#            if np.abs(y_error / Y[test_idx]) > 0.8:
#                print test_idx, ybar, Y[test_idx], newX_test
            er.append(y_error)
        mre = np.mean(np.abs(er)) / np.mean(Y)
        print "MVL with linear fusion function MRE: {0}".format(mre)
#        self.visualize_prediction_error(er, Y, "MVL linear combination")
        
            
        
        
    def visualize_prediction_error(self, er, Y, title):
        cas = Tract.createAllCAObjects()
        import matplotlib.pyplot as plt
        import descartes
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for k in cas:
            re = er[k-1] / Y[k-1]
            if re > 0.4:
                c = 'r'
            elif re < -0.4:
                c = 'b'
            else:
                c = 'w'
            cak = cas[k].polygon
            ax.add_patch(descartes.PolygonPatch(cak, fc=c))
            ax.annotate(str(k), [cak.centroid.x, cak.centroid.y])
        ax.axis('equal')
        ax.set_title(title)
        fig.show()
        
if __name__ == '__main__':
    
    unittest.main()