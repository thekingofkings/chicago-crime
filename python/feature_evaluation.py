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

import sys
import numpy as np
from FeatureUtils import retrieve_crime_count, generate_corina_features, \
    generate_geographical_SpatialLag_ca, generate_GWR_weight, get_centroid_ca
from foursquarePOI import getFourSquarePOIDistribution
from taxiFlow import getTaxiFlow, taxi_flow_normalization
import statsmodels.api as sm
import unittest

N = 77


def extract_raw_samples(year=2010, crime_t=['total'], crime_rate=True):
    """
    Extract all samples with raw labels and features. Return None if the 
    corresponding feature is not selected.
    
    This function is called once only to avoid unnecessary disk I/O.
    
    Input:
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
    assert(Y.shape == (N,1))
    
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



def build_nodal_features( X, leaveOneOut):
    """
    Build nodal features for various prediction models.
    
    Input:
    X - a tuple of nodal features, e.g. (D, ) or (P, ), or (D, P)
    leaveOneOut - the index of testing region
        
    Output:
    Xn - nodal feature vectors. Guaranteed none empty.
    Xn is a (train, test) tuple.
    """
    if leaveOneOut > -1:
        Xn = np.ones((76, 1))
        Xn_test = [1]
        for nodal_feature in X:
            nodal_feature_loo = np.delete(nodal_feature, leaveOneOut, 0)    
            Xn = np.concatenate((Xn, nodal_feature_loo), axis=1)
            Xn_test = np.concatenate((Xn_test, nodal_feature[leaveOneOut, :]))
        assert Xn.shape[0] == 76
    return Xn, Xn_test 



def build_taxi_features(Y, Tf, leaveOneOut, normalization="bydestination"):
    """
    Build taxi flow features.
    
    Input:
    Y - crime rate / count
    Tf - taxi flow count matrix. need normalization first.
    leaveOneOut - the index of testing region
    
    Output:
    T - taxi flow feature, calculated by
            T = Tf * Y
    T is a (train, test) tuple.
    """
    # leave one out
    if leaveOneOut > -1:
        Tf_loo = np.delete(Tf, leaveOneOut, 0)
        Tf_loo = np.delete(Tf_loo, leaveOneOut, 1)
        Y_loo = np.delete(Y, leaveOneOut, 0)
        
    # taxi flow normalization (by destination)
    Tf_norml = taxi_flow_normalization(Tf_loo, normalization)
    
    # calculate taxi flow feature
    T = np.dot(Tf_norml, Y_loo)

    # for testing sample
    if normalization == "bydestination":
        Tf_test = Tf[:,leaveOneOut]
    elif normalization == "bysource" or normalization == "none":
        Tf_test = Tf[leaveOneOut,:]
    Tf_test = np.delete(Tf_test, leaveOneOut).astype(float)
    if normalization != "none":
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



def build_features(Y, D, P, Tf, Yt, Gd, Yg, testK, features=['all'], taxi_norm="bydestination"):
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
    features    - a list features. ['all'] == ['demo', 'poi', 'geo', 'taxi']
    
    Output:
    X_train
    X_test
    Y_train
    Y_test
    """
    X = []
    if 'all' in features or 'demo' in features:
        X.append(D)
    if 'all' in features or 'poi' in features:
        X.append(P)
    Xn = build_nodal_features(tuple(X), testK)
    X_train = Xn[0]
    X_test = Xn[1]
    
    if 'all' in features or 'taxi' in features:
        T = build_taxi_features(Yt, Tf, testK, taxi_norm)
        X_train = np.concatenate((X_train, T[0]), axis=1)
        X_test = np.concatenate((X_test, T[1]))
        
    if 'all' in features or 'geo' in features:
        G = build_geo_features(Yg, Gd, testK)
        X_train = np.concatenate((X_train, G[0]), axis=1)
        X_test = np.concatenate((X_test, G[1]))
    Y_train = np.delete(Y, testK)
    Y_test = Y[testK, 0]
    return X_train, X_test, Y_train, Y_test
    

def leaveOneOut_error(Y, D, P, Tf, Yt, Gd, Yg, features=['all'], gwr_gamma=None, taxi_norm="bydestination"):
    """
    Use GLM model from python statsmodels library to fit data.
    Evaluate with leave-one-out setting, return the average of n errors.
    
    Input:    
    features    - a list features. ['all'] == ['demo', 'poi', 'geo', 'taxi']
    gwr_gamma   - the GWR weight matrx

    Output:
    error - the average error of k leave-one-out evaluation
    """
    errors = []
    for k in range(len(Y)):
        X_train, X_test, Y_train, Y_test = build_features(Y, D, P, Tf, Yt, Gd, Yg, k, features, taxi_norm)
        gamma = np.delete(gwr_gamma[:,k], k) if gwr_gamma is not None else None
        # Train NegativeBinomial Model from statsmodels library
        nbm = sm.GLM(Y_train, X_train, family=sm.families.NegativeBinomial(), freq_weights=gamma)
        nb_res = nbm.fit()
        ybar = nbm.predict(nb_res.params, X_test)
        y_error = np.abs(ybar - Y_test)
        if y_error > 20 * Y_test:
            print k, y_error, Y_test
            continue
        errors.append(y_error)
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
        assert D.shape == (N, 8)
        assert P.shape[0] == N
        assert Tf.shape == (N, N)
        assert Gd.shape == (N, N)
        
    def test_permute_feature(self):
        Y, D, P, Tf, Gd = extract_raw_samples()
        Yp = permute_feature(Y)
        assert np.sum(Yp - Y) != 0
        Dp = permute_feature(D)
        np.testing.assert_almost_equal(np.sum(Dp, axis=0)[1], np.sum(D, axis=0)[1])

    def test_build_nodal_features(self):
        Y, D, P, T, Gd = extract_raw_samples()
        Xn, Xn_test = build_nodal_features((D, P), 3)
        assert Xn_test.shape[0] == Xn.shape[1] 
    
    def test_build_taxi_features(self):
        Y = np.array([1,2,3]).reshape((3,1))
        Tf = np.arange(9).reshape((3,3))
        # normalize by destination
        T, T_test = build_taxi_features(Y, Tf, 1, "bydestination")
        np.testing.assert_almost_equal(T[0,0], 3.0)
        np.testing.assert_almost_equal(T[1,0], 2.6)
        np.testing.assert_almost_equal(T_test[0], 22.0/8)
        # normalize by source
        T, T_test = build_taxi_features(Y, Tf, 1, "bysource")
        np.testing.assert_almost_equal(T[0,0], 3)
        np.testing.assert_almost_equal(T[1,0], 15.0/7)
        np.testing.assert_almost_equal(T_test[0], 18.0/8)
        # without normalize
        T, T_test = build_taxi_features(Y, Tf, 1, "none")
        np.testing.assert_almost_equal(T[0,0], 6)
        np.testing.assert_almost_equal(T[1,0], 30)
        np.testing.assert_almost_equal(T_test[0], 18)

    def test_build_geo_features(self):
        Y = np.array([1,2,3]).reshape((3,1))
        Gd = np.arange(9).reshape((3,3))
        G, G_test = build_geo_features(Y, Gd, 1)
        assert G[0,0] == 6 and G[1,0] == 30 and G_test[0] == 18

    def test_leaveOneOut_error(self):
        Y, D, P, Tf, Gd = extract_raw_samples()
        mae, mre = leaveOneOut_error(Y, D, P, Tf, Y, Gd, Y)
        assert mae <= 1000 and mre < 0.35




def main_evaluate_different_years(year):
    import pickle
    Y, D, P, Tf, Gd = extract_raw_samples(year, crime_t=['total'])
    Yh = pickle.load(open("chicago-hourly-crime-{0}.pickle".format(year)))
    Yh = Yh / D[:,0] * 10000
    assert Yh.shape == (24, N)
    MAE =[]
    MRE = []
    for h in range(24):
        Tf = getTaxiFlow(filename="/taxi-CA-h{0}.matrix".format(h))
        mae, mre = leaveOneOut_error(Yh[h,:].reshape((N,1)), D, P, Tf, Yh[h,:].reshape((N,1)), Gd, 
                                     Yh[h,:].reshape((N,1)), features=['demo', 'poi', 'geo', 'taxi'],
                                       taxi_norm="none")
        print h, mae, mre
        MAE.append(mae)
        MRE.append(mre)
    print year, h, np.mean(MAE), np.mean(MRE)
    with open("kdd16-eval-{0}.pickle".format(year), "w") as fout:
        pickle.dump(MAE, fout)
        pickle.dump(MRE, fout)
    
    
def main_calculate_significance():
    sig = {}
    for f in ["demo", "geo", "taxi", "poi"]:
        Y, D, P, Tf, Gd = extract_raw_samples(2010, crime_t=['total'])
        s = permutation_test_significance(Y, D, P, Tf, Gd, 2000, to_permute=f)
        sig[f] = s
    import pickle
    pickle.dump(sig, open("significance", 'w'))
    
    
def main_evaluate_feature_setting(year=2010, crime_t=['total']):
    feature_settings = [['demo'], ['demo', 'poi'], ['demo', 'taxi'], ['demo', 'poi', 'taxi'],
                        ['demo', 'geo'], ['demo', 'geo', 'poi'], ['demo', 'geo', 'taxi'], ['all']]
    Y, D, P, Tf, Gd = extract_raw_samples(year, crime_t)
    H = [0.08, 0.09, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    nb_MAEs = []
    nb_MREs = []
    gwnbr_MAEs = []
    gwnbr_MREs = []
    for feature_setting in feature_settings:
        mae, mre = leaveOneOut_error(Y, D, P, Tf, Y, Gd, Y, feature_setting)
        nb_MAEs.append(mae)
        nb_MREs.append(mre)
        # Tune bandwidth for GWR model
        gwr_mae = sys.maxint
        gwr_mre = 1.0
        for h in H:
            gwr_gamma = generate_GWR_weight(h)
            mae, mre = leaveOneOut_error(Y, D, P, Tf, Y, Gd, Y, feature_setting, gwr_gamma)
            if mae < gwr_mae:
                gwr_mae = mae
                gwr_mre = mre
        gwnbr_MAEs.append(gwr_mae)
        gwnbr_MREs.append(gwr_mre)
    print "Settings\t",
    for f in feature_settings:
        feature_header = [ele[0] for ele in f]
        print '+'.join(feature_header) + "\t",
    print ""
    print "NB_MAE\t" + '\t'.join(map(str, nb_MAEs))
    print "NB_MRE\t" + '\t'.join(map(str, nb_MREs))
    print "GWNBR_MAE\t" + '\t'.join(map(str, gwnbr_MAEs))
    print "GWNBR_MRE\t" + '\t'.join(map(str, gwnbr_MREs))



def main_evaluate_feature_setting_by_year():
    for year in range(2010, 2015):
        print year
        main_evaluate_feature_setting(year)
    


def main_evaluate_feature_setting_by_type():
    crime_cats = ['THEFT', 'BATTERY', 'NARCOTICS', 'CRIMINAL DAMAGE', 'BURGLARY', 
                  'OTHER OFFENSE', 'ASSAULT', 'MOTOR VEHICLE THEFT', 'ROBBERY',
                  'DECEPTIVE PRACTICE']
    for crime_t in crime_cats:
        print crime_t
        main_evaluate_feature_setting(2010, [crime_t])
    


def main_compare_taxi_normalization_method():    
    H = [0.08, 0.09, 0.1, 0.15, 0.2, 0.3, 0.5]
    for year in range(2010, 2015):
        print year
        Y, D, P, Tf, Gd = extract_raw_samples(year, ["total"])
        for taxi_norm in ["bydestination", "bysource", "none"]:
            gwr_mae = sys.maxint
            gwr_mre = 1.0
            best_h = 0
            for h in H:
                gwr_gamma = generate_GWR_weight(h)
                mae, mre = leaveOneOut_error(Y, D, P, Tf, Y, Gd, Y, ["all"], gwr_gamma, taxi_norm)
                if mae < gwr_mae:
                    gwr_mae = mae
                    gwr_mre = mre
                    best_h = h
            print taxi_norm, gwr_mae, gwr_mre, best_h
        


def ordinary_kriging_evaluation(year):
    """
    Under leave-one-out setting, use only crime rate.
    """
    from pykrige.ok import OrdinaryKriging
    from sklearn.model_selection import LeaveOneOut

    y_cnt = retrieve_crime_count(year)
    demo = generate_corina_features()
    population = demo[1][:,0].reshape(demo[1].shape[0], 1)
    Y = y_cnt / population * 10000
    
    coords = get_centroid_ca()
    
    data = np.concatenate((coords, Y), axis=1)
    loo = LeaveOneOut()
    
    errors = []
    for train_idx, test_idx in loo.split(data):
        x_train = data[train_idx,:]
        coords_test = data[test_idx, [0,1]]
        y_test = data[test_idx, 2]
        
        OK = OrdinaryKriging(x_train[:,0], x_train[:,1], x_train[:,2], variogram_model="linear")
        z, var = OK.execute("points", coords_test[0], coords_test[1])
        errors.append(abs(z[0] - y_test[0]))
    print np.mean(errors), np.mean(errors) / np.mean(Y)
    return errors
    

def regression_kriging_evaluation(year, features_=['all']):
    from pykrige.rk import RegressionKriging
    from sklearn.model_selection import LeaveOneOut
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    svr_model = SVR(kernel='rbf', C=10, gamma=0.001)
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=2)
    Y, D, P, Tf, Gd = extract_raw_samples(year, crime_t=['total'])
    
    coords = get_centroid_ca()
    
    errors = []
    for k in range(77):
        X_train, X_test, Y_train, Y_test = build_features(Y, D, P, Tf, Y, Gd, Y, k, features=features_, taxi_norm="bydestination")
        if k == 0:
            print X_train.shape
        coords_train = np.delete(coords, k, axis=0)
        coords_test = np.array(coords)[k,None]
        m_rk = RegressionKriging(regression_model=rf_model)
        m_rk.fit(X_train, coords_train, Y_train)
        z = m_rk.predict(X_test, coords_test)
        errors.append(abs(Y_test - z[0]))
    print np.mean(errors), np.mean(errors)/np.mean(Y)
    return errors
    

def cokriging_evaluation(year):
    from pyKriging import coKriging
    Y, D, P, Tf, Gd = extract_raw_samples(2012, crime_t=['total'])
    coords = get_centroid_ca()

    X_train, X_test, Y_train, Y_test = build_features(Y, D, P, Tf, Y, Gd, Y, 0, taxi_norm="bydestination")
    coords_train = np.delete(coords, 0, axis=0)
    coKriging.coKriging(coords_train, X_train, coords_train, Y_train)
    


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestFeatureSignificance)
        unittest.TextTestRunner(verbosity=2).run(suite)
    else:
#        main_evaluate_different_years(sys.argv[1])
        # d = ordinary_kriging_evaluation(sys.argv[1])
        feature_comb = [['demo', 'geo'], ['demo', 'geo', 'poi'], ['demo', 'geo', 'taxi'], ['all']]
        for features_ in feature_comb:
            print features_
            d = regression_kriging_evaluation(sys.argv[1], features_)
#    main_calculate_significance()
#        main_evaluate_feature_setting_by_type()
#        main_compare_taxi_normalization_method()
