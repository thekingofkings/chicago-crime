# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:34:01 2016

@author: Hongjian

## Background

This script is created for the KDD and TKDE draft.


## Introduction

Permutation test to calcualte the significance of features in various models, 
such as NB and GWNBR.


## Default Settings

* Region level: CA (we don't consider tract level for now).
* Features of interest: Demo, POI, Geo, Taxi
* Temporal factor is not considered.
"""


import numpy as np
from FeatureUtils import retrieve_crime_count, generate_corina_features, generate_geographical_SpatialLag_ca
from foursquarePOI import getFourSquarePOIDistribution
from taxiFlow import getTaxiFlow
import statsmodels.api as sm
import unittest


def extract_raw_samples(year=2010, crime_t=['total'], crime_rate=True):
    """
    Extract all samples with raw labels and features. Return None if the 
    corresponding feature is not selected.
    
    This function is called once only to avoid unnecessary disk I/O.
    
    Input:
    features    - a list features. ['all'] == ['demo', 'poi', 'geo', 'taxi']
    year        - which year to study
    crime_t     - crime types of interest, e.g. 'total'
    crime_rate  - predict crime_rate or not (count)
    
    Output:
    Y - crime rate / count
    D - demo feature
    P - POI feature
    Tf - taxi flow matrix (count)
    Gd - geo weight matrix
    """
    # Crime count
    y_cnt = retrieve_crime_count(year, col = crime_t)
    
    # Crime rate / count
    demo = generate_corina_features()
    population = demo[1][:,0].reshape(demo[1].shape[0], 1)
    Y = y_cnt / population * 10000 if crime_rate else y_cnt
    assert(Y.shape == (77,1))
    
    # Demo features
    D = demo[1]
    
    # POI features
    P = getFourSquarePOIDistribution(useRatio=False)
    
    # Taxi flow matrix
    Tf = getTaxiFlow(normalization="none")
    
    # Geo weight matrix
    Gd = generate_geographical_SpatialLag_ca()
    
    return Y, D, P, Tf, Gd
    
    

def permute_feature(features):
    """
    Permute features column-wise.
    
    Input:
    features - the feature matrix for multiple instances. Each row is one 
                instance. Each column is one feature.
    Output:
    features_permuted - the permuted features. Each feature is shuffled within
                        column.
    """
    features_permuted = np.copy(features)
    np.random.shuffle(features_permuted)
    return features_permuted



def build_nodal_features(D, P, leaveOneOut):
    """
    Build nodal features for various prediction models.
    
    Input:
    D - the demo feature
    P - the POI feature
    leaveOneOut - the index of testing region
        
    Output:
    Xn - nodal feature vectors
    Xn is a (train, test) tuple.
    """
    if leaveOneOut > -1:
        Xn = np.ones((D.shape[0]-1, 1))
        D_loo = np.delete(D, leaveOneOut, 0)
        P_loo = np.delete(P, leaveOneOut, 0)
        Xn = np.concatenate((Xn, D_loo, P_loo), axis=1)
        assert Xn.shape[0] == 76
    Xn_test = np.concatenate(([1], D[leaveOneOut,:], P[leaveOneOut,:]))
    return Xn, Xn_test 



def build_taxi_features(Y, Tf, leaveOneOut):
    """
    Build taxi flow features.
    
    Input:
    Y - crime rate / count
    Tf - taxi flow count matrix. need normalization first.
    leaveOneOut - the index of testing region
    
    Output:
    T - taxi flow feature, calculated by
            T = Tf' * Y
    T is a (train, test) tuple.
    """
    # leave one out
    if leaveOneOut > -1:
        Tf_loo = np.delete(Tf, leaveOneOut, 0)
        Tf_loo = np.delete(Tf_loo, leaveOneOut, 1)
        Y_loo = np.delete(Y, leaveOneOut, 0)
        
    # taxi flow normalization (by destination)
    Tf_sum = np.sum(Tf_loo, axis=0, keepdims=True).astype(float)
    Tf_sum[Tf_sum==0] = 1
    assert Tf_sum.shape == (1, Y_loo.shape[0])
    Tf_norml = Tf_loo / Tf_sum
    
    # calculate taxi flow feature
    T = np.dot(np.transpose(Tf_norml), Y_loo)

    # for testing sample
    Tf_test = Tf[:,leaveOneOut]
    Tf_test = np.delete(Tf_test, leaveOneOut).astype(float)
    Tf_test /= np.sum(Tf_test) if np.sum(Tf_test) != 0 else 1
    return T, np.dot(Tf_test, Y_loo)
        
    

def build_geo_features(Y, Gd, leaveOneOut=-1):
    """
    Build geo distance weighted features.

    Input:
    Y - crime rate / count
    Gd - geospatial distance weight matrix
    leaveOneOut - the index of testing region

    Output:
    G - geospatial feature, calculated by
            G = Gd * Y
    G is a (train, test) tuple.
    """
    if leaveOneOut > -1:
        Gd_loo = np.delete(Gd, leaveOneOut, 0)
        Gd_loo = np.delete(Gd_loo, leaveOneOut, 1)
        Y_loo = np.delete(Y, leaveOneOut, 0)
    else:
        Gd_loo = Gd
        Y_loo = Y

    Gd_test = np.delete(Gd[leaveOneOut, :], leaveOneOut)
    return np.dot(Gd_loo, Y_loo), np.dot(Gd_test, Y_loo)



def build_features(Y, D, P, Tf, Yt, Gd, Yg, testK):
    """
    Build features for both training and testing samples in leave one out setting.

    Input:
    Y - crime rate/count
    D - demo feature
    P - POI feature
    Tf - taxi flow matrix
    Yt - crime vector for taxi flow calculation
    Gd - geo weight matrix
    Yg - crime vector for geo feature calculation
    testK - index of testing sample
    
    Output:
    X_train
    X_test
    Y_train
    Y_test
    """
    Xn = build_nodal_features(D, P, testK)
    T = build_taxi_features(Yt, Tf, testK)
    G = build_geo_features(Yg, Gd, testK)
    X_train = np.concatenate((Xn[0], T[0], G[0]), axis=1)
    X_test = np.concatenate((Xn[1], T[1], G[1]))
    Y_train = np.delete(Y, testK)
    Y_test = Y[testK, 0]
    return X_train, X_test, Y_train, Y_test
    

def leaveOneOut_error(Y, D, P, Tf, Yt, Gd, Yg):
    """
    Use GLM model from python statsmodels library to fit data.
    Evaluate with leave-one-out setting, return the average of n errors.

    Output:
    error - the average error of k leave-one-out evaluation
    """
    errors = []
    for k in range(len(Y)):
        X_train, X_test, Y_train, Y_test = build_features(Y, D, P, Tf, Yt, Gd, Yg, k)
        # Train NegativeBinomial Model from statsmodels library
        nbm = sm.GLM(Y_train, X_train, family=sm.families.NegativeBinomial())
        nb_res = nbm.fit()
        ybar = nbm.predict(nb_res.params, X_test)
        errors.append(np.abs(ybar - Y_test))
    return np.mean(errors), np.mean(errors) / np.mean(Y)



def permutation_test_significance(Y, D, P, Tf, Gd, n, to_permute="demo"):
    """
    Permutation test on selected features to return significance.
    """
    model_error = leaveOneOut_error(Y, D, P, Tf, Y, Gd, Y)
    Yt = np.copy(Y)
    Yg = np.copy(Y)
    cnt = 0.0
    for i in range(n):
        if i % (n/10) == 0:
            print "Permutation test {0}%, current pvalue {1} ...".format(i/(n/10)*10, cnt/n)
        if to_permute == "demo":
            D = permute_feature(D)
        elif to_permute == "poi":
            P = permute_feature(P)
        elif to_permute == "taxi":
            Yt = permute_feature(Yt)
        elif to_permute == "geo":
            Yg = permute_feature(Yg)
        else:
            raise ValueError("Feature to_permute not found.")
        perm_error = leaveOneOut_error(Y, D, P, Tf, Yt, Gd, Yg)
        if perm_error < model_error:
            cnt += 1
    print "Significance for {0} is {1} with {2} permutations.".format(to_permute, cnt/n, n)
    return cnt / n



class TestFeatureSignificance(unittest.TestCase):

    def test_extract_raw_samples(self):
        Y, D, P, Tf, Gd = extract_raw_samples()
        assert Y is not None
        assert D.shape == (77, 8)
        assert P.shape[0] == 77
        assert Tf.shape == (77, 77)
        assert Gd.shape == (77, 77)
        
    def test_permute_feature(self):
        Y, D, P, Tf, Gd = extract_raw_samples()
        Yp = permute_feature(Y)
        assert np.sum(Yp - Y) != 0
        Dp = permute_feature(D)
        np.testing.assert_almost_equal(np.sum(Dp, axis=0)[1], np.sum(D, axis=0)[1])

    def test_build_nodal_features(self):
        Y, D, P, T, Gd = extract_raw_samples()
        Xn, Xn_test = build_nodal_features(D, P, 3)
        assert Xn_test.shape[0] == Xn.shape[1] 
    
    def test_build_taxi_features(self):
        Y = np.array([1,2,3]).reshape((3,1))
        Tf = np.arange(9).reshape((3,3))
        T, T_test = build_taxi_features(Y, Tf, 1)
        np.testing.assert_almost_equal(T[0,0], 3.0)
        np.testing.assert_almost_equal(T[1,0], 2.6)
        np.testing.assert_almost_equal(T_test[0], 22.0/8)

    def test_build_geo_features(self):
        Y = np.array([1,2,3]).reshape((3,1))
        Gd = np.arange(9).reshape((3,3))
        G, G_test = build_geo_features(Y, Gd, 1)
        assert G[0,0] == 6 and G[1,0] == 30 and G_test[0] == 18

    def test_leaveOneOut_error(self):
        Y, D, P, Tf, Gd = extract_raw_samples()
        mae, mre = leaveOneOut_error(Y, D, P, Tf, Y, Gd, Y)
        assert mae <= 1000 and mre < 0.35




def main_evaluate_different_years():
    for year in range(2010, 2015):
        Y, D, P, Tf, Gd = extract_raw_samples(year, crime_t=['total'])
        mae, mre = leaveOneOut_error(Y, D, P, Tf, Y, Gd, Y)
        print year, mae, mre
    

if __name__ == '__main__':
#    unittest.main()
#    main_evaluate_different_years()
    sig = {}
    for f in ["demo", "geo", "taxi", "poi"]:
        Y, D, P, Tf, Gd = extract_raw_samples(2010, crime_t=['total'])
        s = permutation_test_significance(Y, D, P, Tf, Gd, 2000, to_permute=f)
        sig[f] = s
    import pickle
    pickle.dump(sig, open("significance", 'w'))