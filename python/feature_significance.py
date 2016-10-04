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



def build_nodal_features(D, P, leaveOneOut=-1):
    """
    Build nodal features for various prediction models.
    
    Input:
    D - the demo feature
    P - the POI feature
    leaveOneOut - the index of testing region
        
    Output:
    Xn - nodal feature vectors
    """
    if leaveOneOut > -1:
        Xn = np.ones((D.shape[0]-1, 1))
        D_loo = np.delete(D, leaveOneOut, 0)
        P_loo = np.delete(D, leaveOneOut, 0)
        Xn = np.concatenate((Xn, D_loo, P_loo), axis=1)
        assert Xn.shape[0] == 76
    else:
        Xn = np.ones((D.shape[0],1))
        Xn = np.concatenate((Xn, D, P), axis=1)
        assert Xn.shape[0] == 77
    return Xn    
    



if __name__ == '__main__':
    Y, D, P, Tf, Gd = extract_raw_samples()