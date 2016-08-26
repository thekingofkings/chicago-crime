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
from statsmodels.discrete.discrete_model import CountModel
import unittest
from scipy.special import gammaln, digamma
import statsmodels.api as sm
from scipy import stats
    

class NegBin(CountModel):
    """
    negative binomial regression
    """
    
    def __init__(self, endog, exog, **kwds):
        super(NegBin, self).__init__(endog, exog, offset=None,
            exposure=None, missing=None, **kwds)
        
        
    def loglike(self, params):
        alpha = params[-1]
        beta = params[:-1]
        mu = np.exp(np.dot(self.exog, beta))
        size = 1 / alpha
        prob = size / (size+mu)
        const = gammaln(size+self.endog) - gammaln(self.endog+1) - gammaln(size)
        ll = const + self.endog*np.log(1-prob) + size*np.log(prob)
        return np.sum(ll)
        
        
    def fit(self, start_params=None, method='bfgs', maxiter = 1000, full_output=1,
            disp=1, callback=None, cov_type='nonrobust', cov_kwds=None, **kwds):
        if start_params == None:
            start_params = sm.Poisson(self.endog, self.exog).fit().params
            start_params = np.append(start_params, 0.1)
            if self.exog.mean() > 1:
                start_params[:-1] = np.ones(self.exog.shape[1]) * np.log(self.endog.mean())  / self.exog.mean()
            else:
                start_params[:-1] = np.ones(self.exog.shape[1]) * np.log(self.endog.mean())
#            print "endog mean:", self.endog.mean(), "log endog mean:", np.log(self.endog.mean())
#            print "exog mean", self.exog.mean()
        mlefit = super(NegBin, self).fit(start_params=start_params, maxiter=maxiter, method=method,
                callback=lambda x:x, **kwds)
            
        mlefit._results.params[-1] = np.exp(mlefit._results.params[-1])
        
        nbinfit = NegativeBinomialResults(self, mlefit._results)
        result = NegativeBinomialResultsWrapper(nbinfit)

        if cov_kwds is None:
            cov_kwds = {}  #TODO: make this unnecessary ?
        result._get_robustcov_results(cov_type=cov_type,
                                    use_self=True, use_t=use_t, **cov_kwds)
        return resul 
                

    def score(self, params, **kwds):
        """
        Gradient of log-likelihood evaluated at params
        """
        alpha = params[-1]
        beta = params[:-1]
        exog = self.exog
        y = self.endog[:,None]
        mu = np.exp(np.dot(self.exog, beta))[:,None]
        a1 = 1 / alpha
        
        dparams = exog*a1*(y-mu)/(a1+mu)
        dalpha = (digamma(a1) - digamma(y+a1) + np.log(1+alpha*mu) + \
                alpha* (y-mu)/(1+alpha*mu)).sum() / alpha**2
        
        return np.r_[dparams.sum(0), dalpha]
        
    
    def hessian(self, params):
        """
        Hessian of NB2 model.
        """
        if self._transparams: # lnalpha came in during fit
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        a1 = 1/alpha
        params = params[:-1]

        exog = self.exog
        y = self.endog[:,None]
        mu = self.predict(params)[:,None]

        # for dl/dparams dparams
        dim = exog.shape[1]
        hess_arr = np.empty((dim+1,dim+1))
        const_arr = a1*mu*(a1+y)/(mu+a1)**2
        for i in range(dim):
            for j in range(dim):
                if j > i:
                    continue
                hess_arr[i,j] = np.sum(-exog[:,i,None] * exog[:,j,None] *
                                       const_arr, axis=0)
        tri_idx = np.triu_indices(dim, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]

        # for dl/dparams dalpha
        da1 = -alpha**-2
        dldpda = np.sum(mu*exog*(y-mu)*da1/(mu+a1)**2 , axis=0)
        hess_arr[-1,:-1] = dldpda
        hess_arr[:-1,-1] = dldpda

        # for dl/dalpha dalpha
        #NOTE: polygamma(1,x) is the trigamma function
        da2 = 2*alpha**-3
        dalpha = da1 * (special.digamma(a1+y) - special.digamma(a1) +
                    np.log(a1) - np.log(a1+mu) - (a1+y)/(a1+mu) + 1)
        dada = (da2 * dalpha/da1 + da1**2 * (special.polygamma(1, a1+y) -
                    special.polygamma(1, a1) + 1/a1 - 1/(a1 + mu) +
                    (y - mu)/(mu + a1)**2)).sum()
        hess_arr[-1,-1] = dada

        return hess_arr
                
    
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
        print "NBreg not converged.", res.params[0]
    return res, mod


    
    
class TestNegBin(unittest.TestCase):
    
    def test_withOnlineSource(self):
        import patsy
        import pandas as pd
        url = 'http://vincentarelbundock.github.com/Rdatasets/csv/COUNT/medpar.csv'
        medpar = pd.read_csv(url)
        y, X = patsy.dmatrices('los~type2+type3+hmo+white', medpar)
        res, mod = negativeBinomialRegression(X, y)
        return y, X, medpar, res
        
        
    def test_onChicagoCrimeData(self):
    
        W = generate_transition_SocialLag(2010)
        Yhat = retrieve_crime_count(2009, ['total'])
        Y = retrieve_crime_count(2010, ['total'])
    #    i = retrieve_income_features()
    #    e = retrieve_education_features()
    #    r = retrieve_race_features()
        C = generate_corina_features()
            
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
        from LinearModel import linearRegression
        linearRegression(f, Y)
        return res
        
if __name__ == '__main__':
    unittest.main()