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
from sklearn.preprocessing import scale
import pickle

import sys
sys.path.append("../")

from graph_embedding import get_graph_embedding_features
from feature_evaluation import extract_raw_samples
from Crime import Tract

import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
N = 77


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
    T = get_graph_embedding_features('taxi-CA-static.vec')
    G = get_graph_embedding_features('geo-CA.vec')
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
        assert(Y.shape == (N,1))
        assert(D.shape == (N,8))
        assert(P.shape == (N,10))
        assert(T.shape == (N,8))
        assert(G.shape == (N,8))
        
        
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
    
            
        
        
        
    def test_simple_concatenation_model(self):
        """
        Test a simple concatenation model.
        
        We concatenate the feature vectors from four different views into one
        vector. Then train a NB model on this concatenated vector `X`.
        """
        Y, D, P, T, G = generate_raw_samples(2013)
        X = np.concatenate((D,P,G), axis=1)
#        assert( X.shape == (N, 34) )
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
        
        
def evaluate_various_flow_features_with_concatenation_model(year, spatial):
    Y, D, P, T, G = extract_raw_samples(int(year))
    Yh = pickle.load(open("../chicago-hourly-crime-{0}.pickle".format(year)))
    with open("CAflowFeatures.pickle") as fin:
        mf = pickle.load(fin)
        line = pickle.load(fin)
        dwt = pickle.load(fin)
        dws = pickle.load(fin)
        hdge = pickle.load(fin)
    
    mf_mre = []
    line_mre = []
    dw_mre = []
    for h in range(24):
        print h
        # MF models
        Tmf = mf[h] # sum([e for e in mf.values()])
        import nimfa
        nmf = nimfa.Nmf(G, rank=4, max_iter=100) #, update="divergence", objective="conn", conn_change=50)
        nmf_fit = nmf()
        src = nmf_fit.basis()
        dst = nmf_fit.coef()
        Gmf = np.concatenate((src, dst.T), axis=1)
        
        if spatial == "nospatial":
            X = np.concatenate((D, P, Tmf), axis=1)
        elif spatial == "onlyspatial":
            X = np.concatenate((D, P, Gmf), axis=1)
        elif spatial == "usespatial":
            X = np.concatenate((D, P, Tmf,Gmf), axis=1)
        mre = leaveOneOut_eval(X, Yh[h,:])
        mf_mre.append(mre)
        print "MF MRE: {0}".format(mre)
        
        # LINE model
        Tline = line[h] # sum([e for e in line.values()])
        Gline = get_graph_embedding_features('geo_all.txt')
        if spatial == "nospatial":
            X = np.concatenate((D, P, Tline), axis=1) 
        elif spatial == "onlyspatial":
            X = np.concatenate((D, P, Gline), axis=1) 
        elif spatial == "usespatial":
            X = np.concatenate((D, P, Tline, Gline), axis=1) 
        mre = leaveOneOut_eval(X, Yh[h,:])
        line_mre.append(mre)
        print "LINE_slotted MRE: {0}".format(mre)
        
        # deepwalk
        if spatial == 'nospatial':
            TGdw = dwt[h] # sum([e for e in dw.values()])
        elif spatial == 'onlyspatial':
            TGdw = dws[h]
        elif spatial == 'usespatial':
            TGdw = hdge[h]
        X = np.concatenate((D, P, TGdw), axis=1)
        mre = leaveOneOut_eval(X, Yh[h,:])
        dw_mre.append(mre)
        print "HDGE MRE: {0}".format(mre)
    
    return mf_mre, line_mre, dw_mre


def evaluate_various_embedding_features_with_lag_model(year, spatial):
    Y, D, P, T, G = extract_raw_samples(int(year))
    population = D[:,0]
    
    Yh = pickle.load(open("../chicago-hourly-crime-{0}.pickle".format(year)))
    Yh = Yh / population
    assert Yh.shape == (24, N)
    
    with open("CAflowFeatures.pickle") as fin:
        mf = pickle.load(fin)
        line = pickle.load(fin)
        dwt = pickle.load(fin)
        dws = pickle.load(fin)
        hdge = pickle.load(fin)
    
    mf_mre = []
    mf_mae = []
    line_mre = []
    line_mae = []
    dw_mre = []
    dw_mae = []
    for h in range(24):
        print h
        # MF models
        Tmf = mf[h] # sum([e for e in mf.values()])
        import nimfa
        nmf = nimfa.Nmf(G, rank=4, max_iter=100) #, update="divergence", objective="conn", conn_change=50)
        nmf_fit = nmf()
        src = nmf_fit.basis()
        dst = nmf_fit.coef()
        Gmf = np.concatenate((src, dst.T), axis=1)
        
        if spatial == "nospatial":
            X = np.concatenate((D, P, Tmf), axis=1)
        elif spatial == "onlyspatial":
            X = np.concatenate((D, P, Gmf), axis=1)
        elif spatial == "usespatial":
            X = np.concatenate((D, P, Tmf, Gmf), axis=1)
        mre, mae = leaveOneOut_eval(X, Yh[h,:].reshape((N,1)))
        mf_mre.append(mre)
        mf_mae.append(mae)
        print "MF MRE: {0}".format(mre)
        
        # LINE model
        Tline = line[h] # sum([e for e in line.values()])
        Gline = get_graph_embedding_features('geo_all.txt')
        if spatial == "nospatial":
            X = np.concatenate((D, P, Tline), axis=1) 
        elif spatial == "onlyspatial":
            X = np.concatenate((D, P, Gline), axis=1) 
        elif spatial == "usespatial":
            X = np.concatenate((D, P, Tline, Gline), axis=1) 
        mre, mae = leaveOneOut_eval(X, Yh[h,:].reshape((N,1)))
        line_mre.append(mre)
        line_mae.append(mae)
        print "LINE_slotted MRE: {0}".format(mre)
        
        # deepwalk
#        TGdw = dw[h] # sum([e for e in dw.values()])
        X = np.concatenate((D, P, hdge[h]), axis=1)
        mre, mae = leaveOneOut_eval(X, Yh[h,:].reshape((N,1)))
        dw_mre.append(mre)
        dw_mae.append(mae)
        print "HDGE MRE: {0}".format(mre)
    
    return mf_mre, line_mre, dw_mre, mf_mae, line_mae, dw_mae




    
def leaveOneOut_eval(X, Y):
    X = sm.add_constant(X, prepend=False)
    loo = LeaveOneOut(len(Y))
    er = []
    for train_idx, test_idx in loo:
        nbm, yp = NBmodel(train_idx, Y, X)
        ybar = nbm.predict(X[test_idx])
        y_error = np.abs(ybar - Y[test_idx])
        if y_error > 20 * Y[test_idx]:
            print test_idx, y_error, Y[test_idx]
            continue
        er.append(y_error)
    max_idx = np.argmax(er)
    print "largest error", er[max_idx], Y[max_idx], max_idx+1
    mae = np.mean(er)
    mre = mae / np.mean(Y)
    return mre, mae
    
if __name__ == '__main__':
    
#    unittest.main()
    import sys
    year = sys.argv[1]
#    r = evaluate_various_flow_features_with_concatenation_model(sys.argv[1], sys.argv[2]) # year and spatial
    r = evaluate_various_embedding_features_with_lag_model(year, sys.argv[2])
    pickle.dump(r, open("embeddings-{0}.pickle".format(year), "w"))
    print np.mean(r, axis=1)
    
    with open("../kdd16-eval-{0}.pickle".format(year)) as fin:
        kdd_mae = pickle.load(fin)
        kdd_mre = pickle.load(fin)
        
    mf_mre = r[0]
    mf_mae = r[3]
    line_mre = r[1]
    line_mae = r[4]
    dge_mre = r[2]
    dge_mae = r[5]
    
    plt.figure()
    plt.plot(mf_mre)
    plt.plot(line_mre)
    plt.plot(kdd_mre)
    plt.plot(dge_mre)
    
    plt.legend(["MF", "LINE", "KDD", "DGE"])
    