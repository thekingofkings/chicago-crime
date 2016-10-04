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
    Y = y_cnt / population if crime_rate else y_cnt
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
    for i in range(features_permuted.shape[1]):
        np.random.shuffle(features_permuted[:,i])
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



def leaveOneOut(Y, D, P, Tf, Gd, leaveOneOut):
    Xn = build_nodal_features(D, P, leaveOneOut)
    T = build_taxi_features(Y, Tf, leaveOneOut)
    G = build_geo_features(Y, Gd, leaveOneOut)
    X = np.concatenate(Xn, T, G)




class TestFeatureSignificance(unittest.TestCase):

    def test_extract_raw_samples(self):
        Y, D, P, Tf, Gd = extract_raw_samples()
        assert Y is not None
        assert D.shape == (77, 8)
        assert P.shape[0] == 77
        assert Tf.shape == (77, 77)
        assert Gd.shape == (77, 77)

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


if __name__ == '__main__':
    unittest.main()
