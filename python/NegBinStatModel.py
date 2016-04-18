# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:51:54 2016

@author: hxw186


Factor out regression model of my own implementation.

My own Negative Binomial implementation relies on the statsmodels python library.
This library is not well-implemented. Therefore, my own NB implementation here
is deprecated.
"""

from FeatureUtils import *
import numpy as np
from scipy.stats import nbinom
from statsmodels.base.model import GenericLikelihoodModel




"""
Part Two
Regression models
"""

  
    

class NegBin(GenericLikelihoodModel):
    """
    negative binomial regression
    """
    
    def __init__(self, endog, exog, **kwds):
        super(NegBin, self).__init__(endog, exog, **kwds)
        
        
    def nloglikeobs(self, params):
        alpha = params[-1]
        beta = params[:-1]
        mu = np.exp(np.dot(self.exog, beta))
        size = 1 / alpha
        prob = size / (size+mu)
      #  ll = 0
        # for idx, y in enumerate(self.endog):
         #    ll += gammaln(y + size) - gammaln(size) - gammaln(y+1) + y * np.log(mu * alpha / (mu *alpha + 1))- size * np.log(mu * alpha + 1)
        ll = nbinom.logpmf( self.endog, size, prob)
        return -ll
        
    def fit(self, start_params=None, maxiter = 10000, maxfun=10000, **kwds):
        if start_params == None:
            start_params = np.append(np.zeros(self.exog.shape[1]), .5)
            if self.exog.mean() > 1:
                start_params[:-1] = np.ones(self.exog.shape[1]) * np.log(self.endog.mean())  / self.exog.mean()
            else:
                start_params[:-1] = np.ones(self.exog.shape[1]) * np.log(self.endog.mean())
#            print "endog mean:", self.endog.mean(), "log endog mean:", np.log(self.endog.mean())
#            print "exog mean", self.exog.mean()
        return super(NegBin, self).fit(start_params=start_params, maxiter=maxiter,
                maxfun=maxfun, **kwds)
                
    
    def predict(self, params, exog=None, *args, **kwargs):
        """
        predict the acutal endogenous count from exog
        """
        beta = params[:-1]
        return np.exp(exog.dot(beta))
  


    
def negativeBinomialRegression(features, Y):
    """
    learn the NB regression
    """
    mod = NegBin(Y, features)
    res = mod.fit(disp=False)
    if not res.mle_retvals['converged']:
        print "NBreg not converged.", res.params[0], ",", res.pvalues[0]
    return res, mod


    
def unitTest_withOnlineSource():
    import patsy
    import pandas as pd
    url = 'http://vincentarelbundock.github.com/Rdatasets/csv/COUNT/medpar.csv'
    medpar = pd.read_csv(url)
    y, X = patsy.dmatrices('los~type2+type3+hmo+white', medpar)
    res = negativeBinomialRegression(X, y)
    return y, X, medpar, res
    
    
def unitTest_onChicagoCrimeData():

    W = generate_transition_SocialLag(2010)
    Yhat = retrieve_crime_count(2009, ['total'])
    Y = retrieve_crime_count(2010, ['total'])
#    i = retrieve_income_features()
#    e = retrieve_education_features()
#    r = retrieve_race_features()
#    C = generate_corina_features()
        
    f1 = np.dot(W, Yhat)
    # f = np.concatenate((f1, i[1], e[1], r[1], np.ones(f1.shape)), axis=1)
    # f = pd.DataFrame(f, columns=['social lag'] + i[0] + e[0] + r[0] + ['intercept'])
#    f = scale(f)
#    f = np.concatenate((f, np.ones(f1.shape)), axis=1)
    f = np.concatenate( (C[1], np.ones(f1.shape)), axis=1 )
    np.savetxt("Y.csv", Y, delimiter=",")
    f = pd.DataFrame(f, columns=C[0] + ['intercept'])
    f.to_csv("f.csv", sep="," )
    Y = Y.reshape((77,))
    print "Y", Y.mean()
    res = negativeBinomialRegression(f, Y)
    
    
    print "f shape", f.shape
    print "Y shape", Y.shape
    linearRegression(f, Y)
    return res