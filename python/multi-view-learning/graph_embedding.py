 
import numpy as np
import statsmodels.api as sm
from sklearn.cross_validation import LeaveOneOut

import sys
sys.path.append("../")

from FeatureUtils import retrieve_crime_count, generate_corina_features




def leaveOneOut_error(Y, X):
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
    errs_train = np.zeros(2)
    loo = LeaveOneOut(len(Y))
    for train_idx, test_idx in loo:
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        # Train NegativeBinomial Model from statsmodels library
        nbm = sm.GLM(Y_train, X_train, family=sm.families.Poisson())
        nb_res = nbm.fit()
        ybar = nbm.predict(nb_res.params, X_train)
        er_train = np.mean(np.abs(ybar - Y_test))
        errs_train += er_train, er_train / np.mean(Y_test)
        print er_train, er_train / np.mean(Y_test)
        
        ybar = nbm.predict(nb_res.params, X_test)
        errors.append(np.abs(ybar - Y_test))
    print errs_train / len(Y)
    return np.mean(errors), np.mean(errors) / np.mean(Y)
    
    
    

if __name__ == '__main__':
    ge = []
    with open('vec_all_20.txt', 'r') as fin:
        header = fin.readline()
        for line in fin:
            ls = line.strip().split(" ")
            ge.append([float(i) for i in ls])
    ge = np.array(ge)
    ge[:,0] = 1
    y_cnt = retrieve_crime_count(2010)
    demo = generate_corina_features()
    population = demo[1][:,0].reshape(demo[1].shape[0], 1)
    y = y_cnt / population * 10000
    
    er = leaveOneOut_error(y, ge)
    