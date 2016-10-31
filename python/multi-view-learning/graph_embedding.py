 
import numpy as np
import statsmodels.api as sm
from sklearn.cross_validation import LeaveOneOut

import sys
sys.path.append("../")

from FeatureUtils import retrieve_crime_count




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
    loo = LeaveOneOut(len(Y))
    for train_idx, test_idx in loo:
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        # Train NegativeBinomial Model from statsmodels library
        nbm = sm.GLM(Y_train, X_train, family=sm.families.Poisson())
        nb_res = nbm.fit()
        print nb_res.params
        ybar = nbm.predict(nb_res.params, X_test)
        errors.append(np.abs(ybar - Y_test))
        print ybar, Y_test, np.abs(ybar - Y_test)
    return np.mean(errors), np.mean(errors) / np.mean(Y)
    
    
    

if __name__ == '__main__':
    ge = []
    with open('vec_all_20.txt', 'r') as fin:
        header = fin.readline()
        for line in fin:
            ls = line.strip().split(" ")
            ge.append([float(i) for i in ls[1:]])
    ge = np.array(ge)
    y = retrieve_crime_count(2010)
    
    er = leaveOneOut_error(y, ge)