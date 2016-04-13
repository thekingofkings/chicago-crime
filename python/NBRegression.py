"""
Run the Negative Binomial Regression model

The regresion model use various features to predict the crime count in each
unit of study area (i.e. tract or Community Area)

Author: Hongjian
date:8/20/2015
"""


"""
Part One
Generate vairous features

factor into separate file FeatureUtils
"""

from FeatureUtils import *
import warnings
warnings.filterwarnings('ignore')


"""
Part Two
Regression models
"""


import numpy as np
from scipy.stats import nbinom
from statsmodels.base.model import GenericLikelihoodModel


"""
Part Three
build model and compare
"""

import pandas as pd
import statsmodels.api as sm
from sklearn import cross_validation

# misc libraries
import matplotlib.pyplot as plt
import subprocess
import os.path
import os
from sklearn.utils import shuffle

here = os.path.dirname(os.path.abspath(__file__))



"""
Part Two
Regression models
"""



def linearRegression(features, Y):
    """
    learn the linear regression model from features to Y
    output the regression analysis parameters
    plot scatter plot
    """
    
    from sklearn import linear_model
#    mod = sm.OLS(Y, features )
    mod = linear_model.LinearRegression()
    res = mod.fit(features, Y)
    return res


    
    

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
     
    




"""
Part Three

Evaluation and fitting real models on Chicago data.
"""

from foursquarePOI import getFourSquarePOIDistribution
from taxiFlow import getTaxiFlow


def leaveOneOut_evaluation_onChicagoCrimeData(year=2010, features= ["all"], 
                                              crime_t=['total'], flow_type=0, 
                                              verboseoutput=False, region='ca',
                                              weightSocialFlow=True, 
                                              useRate=True, logFeatures = []):
    """
    Generate the social lag from previous year
    use income/race/education of current year
    """
    if 'sociallag' in features:
        W = generate_transition_SocialLag(year, lehd_type=flow_type, region=region,
                                          normalization='pair')
    
    
    # add POI distribution and taxi flow
    poi_dist = getFourSquarePOIDistribution(useRatio=False, gridLevel=region)
    F_taxi = getTaxiFlow(normalization="bydestination", gridLevel=region)
        
        
    if region == 'ca':
        W2 = generate_geographical_SpatialLag_ca()
        
        Yhat = retrieve_crime_count(year-1, col = crime_t)
#        h = retrieve_health_data()
#        Y = h[0].reshape((77,1))
        Y = retrieve_crime_count(year, col = crime_t)
        C = generate_corina_features()
        popul = C[1][:,0].reshape(C[1].shape[0],1)
        
        
        if 'sociallag' in features:
            """ use poverty demographics to weight social lag """
            wC = 28 # 130.0 if useRate else 32.0     # constant parameter
            if weightSocialFlow:
                poverty = C[1][:,2]        
                for i in range(W.shape[0]):
                    for j in range (W.shape[1]):
                        W[i][j] *= np.exp( - np.abs(poverty[i] - poverty[j]) / wC )
        
        # crime count is normalized by the total population as crime rate
        # here we use the crime count per 10 thousand residents
        if useRate:
            Y = np.divide(Y, popul) * 10000
            Yhat = np.divide(Yhat, popul) * 10000
    elif region == 'tract':
        W2, tractkey = generate_geographical_SpatialLag()
    
        Yhat_map = retrieve_crime_count(year-1, col = crime_t, region='tract')
        Yhat = np.array( [Yhat_map[k] for k in tractkey] ).reshape( len(Yhat_map), 1)
        
        Y_map = retrieve_crime_count(year, col = crime_t, region='tract')
        Y = np.array( [Y_map[k] for k in tractkey] ).reshape( len(Y_map), 1 )
        
        C = generate_corina_features(region='tract')
        C_mtx = []
        cnt = 0
        
        for k in tractkey:
            if k in C[1]:
                C_mtx.append(C[1][k])
            else:
                cnt += 1
                C_mtx.append( [0 for i in range(7)] )
        
        C = ( C[0], np.array( C_mtx ) )
        
        
        # at tract level we don't normalize by population, since the tract is
        # defined as region with around 2000 population
        if useRate:
            pass
    
    
    
    i = retrieve_income_features()
    e = retrieve_education_features()
    r = retrieve_race_features()
    
    f2 = np.dot(W2, Y)
    ftaxi = np.dot(F_taxi, Y)
    
    
    # add intercept
    columnName = ['intercept']
    f = np.ones(f2.shape)
    lrf = np.copy(f)

    if "all" in features:
        f = np.concatenate( (f, f1, i[1], e[1], r[1]), axis=1)
        f = pd.DataFrame(f, columns=['social lag'] + i[0] + e[0] + r[0])
    if "sociallag" in features:        
        f1 = np.dot(W, Y)
        if 'sociallag' in logFeatures:
            f = np.concatenate( (f, np.log(f1)), axis=1 )
        else:
            f = np.concatenate( (f, f1), axis=1)
        lrf = np.concatenate( (f, f1), axis=1)
        columnName += ['social lag']
    if  "income" in features:
        f = np.concatenate( (f, i[1]), axis=1)
        lrf = np.concatenate( (f, i[1]), axis=1)
        columnName += i[0]
    if "race" in features:
        f = np.concatenate( (f, r[1]), axis=1)
        lrf = np.concatenate( (f, r[1]), axis=1)
        columnName += r[0]
    if "education" in features :
        f = np.concatenate( (f, e[1]), axis=1)
        lrf = np.concatenate( (f, e[1]), axis=1)
        columnName += e[0]
    if 'corina' in features :
        f = np.concatenate( (f, C[1]), axis=1)
        lrf = np.concatenate( (f, C[1]), axis=1)
        columnName += C[0]
    if 'spatiallag' in features:
        if 'spatiallag' in logFeatures:
            f = np.concatenate( (f, np.log(f2)), axis=1)
        else:
            f = np.concatenate( (f, f2), axis=1)
        lrf = np.concatenate( (f, f2), axis=1)
        columnName += ['spatial lag']
    if 'taxiflow' in features:
        if 'taxiflow' in logFeatures:
            f = np.concatenate( (f, np.log(ftaxi)), axis=1 )
        else:
            f = np.concatenate( (f, ftaxi), axis=1 )
        lrf = np.concatenate( (f, ftaxi), axis=1 )
        columnName += ['taxi flow']
    if 'POIdist' in features:
        f = np.concatenate( (f, poi_dist), axis=1 )
        lrf = np.concatenate( (f, poi_dist), axis=1 )
        columnName += ['POI food', 'POI residence', 'POI travel', 'POI arts entertainment', 
                       'POI outdoors recreation', 'POI education', 'POI nightlife', 
                       'POI professional', 'POI shops', 'POI event']

    
    if 'temporallag' in features:
        f = np.concatenate( (f, np.log(Yhat)), axis=1)
        lrf = np.concatenate( (f, Yhat), axis=1)
        columnName += ['temporal lag']  
    f = pd.DataFrame(f, columns = columnName)
    

    # call the Rscript to get Negative Binomial Regression results
    np.savetxt("Y.csv", Y, delimiter=",")
    f.to_csv("f.csv", sep=",", index=False)
    if verboseoutput:
        subprocess.call( ['Rscript', 'nbr_eval.R', region, 'verbose'] )
    else:
        nbres = subprocess.check_output( ['Rscript', 'nbr_eval.R', region] )
    
    Y = Y.reshape((len(Y),))
    loo = cross_validation.LeaveOneOut(len(Y))
#    mae = 0
    mae2 = 0
#    errors1 = []
    errors2 = []
    for train_idx, test_idx in loo:
        f_train, f_test = lrf[train_idx], lrf[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
#        res, mod = negativeBinomialRegression(f_train, Y_train)
#        ybar = mod.predict(res.params, exog=f_test)
#        errors1.append( np.abs(Y_test - ybar.values[0])[0] )
        
        if not np.any(np.isnan(f_train)) and np.all(np.isfinite(f_train)):
            r2 = linearRegression(f_train, Y_train)
            y2 = r2.predict(f_test)
            errors2.append( np.abs( Y_test - y2 ) )
    #        print test_idx, Y_test[0], ybar.values[0], y2[0]
            if verboseoutput:
                print Y_test[0], y2[0]
        else:
            print 'nan or infinite'
            pass
        
        
#    mae = np.mean(errors1)
    mae2 = np.mean(errors2)
#    var = np.sqrt( np.var(errors1) )
    var2 = np.sqrt( np.var(errors2) )
#    mre = mae / Y.mean()
    mre2 = mae2 / Y.mean()
#    print "NegBio Regression MAE", mae, "std", var, "MRE", mre
    if verboseoutput:
        print "Linear Regression MAE", mae2, "std", var2, "MRE", mre2
    else:
        print nbres
        print mae2, var2, mre2
        return np.array([[float(ele) for ele in nbres.split(" ")], [mae2, var2, mre2]])
    



def tenFoldCV_onChicagoCrimeData(features=['corina'], CVmethod='10Fold', P = 10, NUM_ITER=20, SHUFFLE=True):
    """
    Use different years data to train the NB model
    """
    YEARS = range(2003, 2014)
    
    Y = []
    C = []
    FL = []
    GL = []
    T = []
    for year in YEARS:
        W = generate_transition_SocialLag(year, lehd_type=0)
        Yhat = retrieve_crime_count(year-1, ['total'])
        y = retrieve_crime_count(year, ['total'])
        c = generate_corina_features()
        popul = c[1][:,0].reshape((77,1))
        
        # crime count is normalized by the total population as crime rate
        # here we use the crime count per 10 thousand residents
        y = np.divide(y, popul) * 10000
        Yhat = np.divide(Yhat, popul) * 10000
        
        W2 = generate_geographical_SpatialLag_ca()
        
        f1 = np.dot(W, Yhat)
        f2 = np.dot(W2, Yhat)
        
        FL.append(f1)
        GL.append(f2)
        Y.append(y)
        T.append(Yhat)
        C.append(c[1])
    
    
    Y = np.concatenate(Y, axis=0)
    columnName = ['intercept']
    f = np.ones(Y.shape)
    if 'corina' in features:
        C = np.concatenate(C, axis=0)
        f = np.concatenate( (f, C), axis=1 )
        columnName += c[0]
    if 'sociallag' in features:
        FL = np.concatenate(FL, axis=0)
        f = np.concatenate( (f, FL), axis = 1)
        columnName += ['sociallag']
    if 'spatiallag' in features:
        GL = np.concatenate(GL, axis=0)
        f = np.concatenate((f, GL), axis=1)
        columnName += ['spatiallag']
    if 'temporallag' in features:
        T = np.concatenate(T, axis=0)
        f = np.concatenate((f, T), axis=1)
        columnName += ['temporallag']
    
    
    
    if SHUFFLE:
        f, Y = shuffle(f, Y)
    
    if CVmethod == '10Fold':
        splt = cross_validation.KFold(n=f.shape[0], n_folds=10, shuffle=True)
    elif CVmethod == 'leaveOneOut':
        splt = cross_validation.LeaveOneOut(n=f.shape[0])
    elif CVmethod == 'leavePOut':
        splt = cross_validation.LeavePOut(n=f.shape[0], p = P)
    
    mae1 = []
    mae2 = []
    mre1 = []
    mre2 = []
    sd_mae1 = []
    sd_mae2 = []
    sd_mre1 = []
    sd_mre2 = []
    med_mae1 = []
    med_mae2 = []
    med_mre1 = []
    med_mre2 = []
    cnt = 0
    
    if CVmethod == 'leaveOneOut':
        y_gnd = []
        y_lr = []


    for train_idx, test_idx in splt:
        cnt += 1
        if cnt > NUM_ITER:
            break
        f_train, f_test = f[train_idx, :], f[test_idx, :]
        Y_train, Y_test = Y[train_idx, :], Y[test_idx, :]
        

        # write file for invoking NB regression in R        
        np.savetxt("Y_train.csv", Y_train, delimiter=",")
        np.savetxt("Y_test.csv", Y_test, delimiter=",")        
        pd.DataFrame(f_train, columns = columnName).to_csv("f_train.csv", sep=",", index=False)
        pd.DataFrame(f_test, columns = columnName).to_csv("f_test.csv", sep=",", index=False)
        
        # NB regression 
        nbres = subprocess.check_output( ['Rscript', 'nbr_eval_kfold.R'] ).split(" ")
        y1 = np.array([float(e) for e in nbres])
        y1 = y1.reshape((y1.shape[0], 1))
        a = np.abs( Y_test - y1 )
        
        mae1.append(np.mean(a))
        sd_mae1.append(np.std(a))
        med_mae1 += a.tolist()
        r = a / Y_test
        mre1.append(np.mean(r))
        sd_mre1.append(np.std(r))
        med_mre1 += r.tolist()
        
        # Linear regression
        r2 = linearRegression(f_train, Y_train)
        y2 = r2.predict(f_test)
        y2 = y2.reshape((y2.shape[0], 1))
        ae = np.abs(Y_test - y2)
        mae2.append( np.mean(ae) )
        sd_mae2.append( np.std(ae) )
        med_mae2 += ae.tolist()
        re = ae / Y_test
        mre2.append( np.mean(re))
        sd_mre2.append( np.std(re) )
        med_mre2 += re.tolist()
        
        if CVmethod == 'leaveOneOut':
            y_gnd.append(Y_test)
            y_lr.append(y2)
    
    
    if CVmethod == 'leaveOneOut':
        print np.mean(mae1), np.median(mae1), np.mean(mre1), np.median(mre1),
        print np.mean(mae2), np.median(mae2), np.mean(mre2), np.median(mre2)
        return y_gnd, y_lr
    else:
        print np.mean(mae1), np.mean(sd_mae1), np.median(med_mae1), np.mean(mre1), np.mean(sd_mre1), np.median(med_mre1),
        print np.mean(mae2), np.mean(sd_mae2), np.median(med_mae2), np.mean(mre2), np.mean(sd_mre2), np.median(med_mre2)
        
    return mae1, mae2





def permutationTest_onChicagoCrimeData(year=2010, features= ["all"], logFeatures = [], flowType=0, 
                                       crimeType = ['total'], iters=1001):
    """
    Permutation test with regression model residuals
    
    How to do the permutation?
    
    Initial try - before 2015/10/4
    
    For each sample point (CA), we permute the dependent variable (crime count),
    while keeps the dependent variables the same.
    
    This approach is hard to explain.
    
    
    Second try:
    permute the feature of interest
    """
    W = generate_transition_SocialLag(year, lehd_type=flowType, normalization='source')
    np.savetxt(here + "/W.csv", W, delimiter="," )

    Yhat = retrieve_crime_count(year-1, crimeType)
    Y = retrieve_crime_count(year, crimeType)
    C = generate_corina_features()
    popul = C[1][:,0].reshape((77,1))
    
    # crime count is normalized by the total population as crime rate
    # here we use the crime count per 10 thousand residents
#    Y = np.divide(Y, popul) * 10000
#    Yhat = np.divide(Yhat, popul) * 10000
    
    W2 = generate_geographical_SpatialLag_ca()
    np.savetxt(here + "/W2.csv", W2, delimiter=",")
    
    i = retrieve_income_features()
    e = retrieve_education_features()
    r = retrieve_race_features()
    
    f1 = np.dot(W, Y)
    f2 = np.dot(W2, Y)
    # add intercept
    columnName = ['intercept']
    f = np.ones(f1.shape)
    flr = np.ones(f1.shape)

    if "all" in features:
        f = np.concatenate( (f, f1, i[1], e[1], r[1]), axis=1)
        flr = np.concatenate( (flr, f1, i[1], e[1], r[1]), axis=1)
        f = pd.DataFrame(f, columns=['social lag'] + i[0] + e[0] + r[0])
        flr = pd.DataFrame(flr, columns=['social lag'] + i[0] + e[0] + r[0])
        features.remove('all')
    if "sociallag" in features: 
        if 'sociallag' in logFeatures:
            f = np.concatenate( (f, np.log(f1)), axis=1 )
        else:
            f = np.concatenate( (f, f1), axis=1)
        flr = np.concatenate( (flr, f1), axis=1)
        columnName += ['social lag']
        features.remove('sociallag')
    if  "income" in features:
        f = np.concatenate( (f, i[1]), axis=1)
        flr = np.concatenate((flr, i[1]), axis=1)
        columnName += i[0]
        features.remove('income')
    if "race" in features:
        f = np.concatenate( (f, r[1]), axis=1)
        flr = np.concatenate( (flr, r[1]), axis=1)
        columnName += r[0]
        features.remove('race')
    if "education" in features :
        f = np.concatenate( (f, e[1]), axis=1)
        flr = np.concatenate((flr, e[1]), axis=1)
        columnName += e[0]
        features.remove('education')
    if 'corina' in features:
        flr = np.concatenate( (flr, C[1]), axis=1)
        C[1][:,0] = np.log(C[1][:,0])
        f = np.concatenate( (f, C[1]), axis=1)
        columnName += C[0]
        features.remove('corina')
    if 'spatiallag' in features:
        if 'spatiallag' in logFeatures:
            f = np.concatenate( (f, np.log(f2)), axis=1 )
        else:
            f = np.concatenate( (f, f2), axis=1)
        flr = np.concatenate( (flr, f2), axis=1 )
        columnName += ['spatial lag']
        features.remove('spatiallag')
    if 'temporallag' in features:
        if 'temporallag' in logFeatures:
            f = np.concatenate( (f, np.log(Yhat)), axis=1 )
        else:
            f = np.concatenate( (f, Yhat), axis=1)
        flr = np.concatenate( (flr, Yhat), axis=1 )
        columnName += ['temporal lag']
        features.remove('temporallag')
    
    # features contains more specific categories of demographics feature
    for demo in features:
        i = C[0].index(demo)
        tmp = C[1][:,i]
        tmp = tmp.reshape((len(tmp), 1))
        if demo in logFeatures:
            f = np.concatenate( (f, np.log(tmp)), axis=1 )
        else:
            f = np.concatenate( (f, tmp), axis=1 )
        flr = np.concatenate( (flr, tmp), axis=1 )
        columnName += [demo]
        
    
    f = pd.DataFrame(f, columns = columnName)
    flr = pd.DataFrame(flr, columns = columnName)
    
    flr.to_csv(here + "/flr.csv", sep=",", index=False)
    f.to_csv(here + "/f.csv", sep=",", index=False)
    np.savetxt(here + "/Y.csv", Y, delimiter=",")
    subprocess.call( ['Rscript', 'permutation_test.R'], cwd=here )
    
    """
    The following permutation design is obsolete, due to the inefficiency.
    
    The permuation now is finished in the Rscript. In this way, we save time
    for repeating I/O on the similar Y.csv and f.csv
    
    
    =================  old code starts here =============
    
    # permute each column
            
        # initialization
        LR_coeffs = []
        if os.path.exists(here + '/coefficients.txt'):
            os.remove(here + '/coefficients.txt')
            
        for i in range(iters):
            if i == 0:
                pidx = range(len(Y))
            else:
                pidx = np.random.permutation(len(Y))
            
            # permute the column
            f[columnKey] = f[columnKey].values[pidx]
            # call the Rscript to get Negative Binomial Regression results         
            f.to_csv(here + "/f.csv", sep=",", index=False)
            np.savetxt(here + "/Y.csv", Y, delimiter=",")
            subprocess.call( ['Rscript', 'nbr_permutation_test.R'], cwd=here )
            
            # LR permutation test
            flr[columnKey] = flr[columnKey].values[pidx]
            lrmod = linearRegression(flr, Y)
            LR_coeffs.append(lrmod.params)
            
        NB_coeffs = np.loadtxt(fname=here + '/coefficients.txt', delimiter=',')
        LR_coeffs = np.array(LR_coeffs)
        
        
        # process columns: distribution of permutations
        
        column = NB_coeffs[:,idx]
        targ = column[0]
        cnt = 0.0
        for e in column:
            if e > targ:
                cnt += 1
        nb_p = cnt / len(column)
                
        lr_col = LR_coeffs[:,idx]
        lr_trg = lr_col[0]
        lr_cnt = 0.0
        for e in lr_col:
            if e > lr_trg:
                lr_cnt += 1
        lr_p = lr_cnt / len(column)       
                
        print targ, nb_p, lr_trg, lr_p

        
        
        plt.figure(figsize=(8,3))
        # NB
        plt.subplot(1,2,1)
        plt.hist(column)
        plt.axvline(x = targ, linewidth=4, color='r')
        plt.title("NB {0} p {1:.4f}".format(columnName[idx], nb_p))
        # LR
        plt.subplot(1,2,2)
        plt.hist(lr_col)
        plt.axvline(x = lr_trg, linewidth=4, color='r')
        plt.title("LR {0} p {1:.4f}".format(columnName[idx], lr_p))
        plt.savefig(here + '/PT-{0}.png'.format(columnKey), format='png')
        
    =================  old code finishes here =============  
    """ 
    
    
    


def crimeRegression_eachCategory(year=2010):
    header = ['ARSON', 'ASSAULT', 'BATTERY', 'BURGLARY', 'CRIM SEXUAL ASSAULT', 
    'CRIMINAL DAMAGE', 'CRIMINAL TRESPASS', 'DECEPTIVE PRACTICE', 
    'GAMBLING', 'HOMICIDE', 'INTERFERENCE WITH PUBLIC OFFICER', 
    'INTIMIDATION', 'KIDNAPPING', 'LIQUOR LAW VIOLATION', 'MOTOR VEHICLE THEFT', 
    'NARCOTICS', 'OBSCENITY', 'OFFENSE INVOLVING CHILDREN', 'OTHER NARCOTIC VIOLATION',
    'OTHER OFFENSE', 'PROSTITUTION', 'PUBLIC INDECENCY', 'PUBLIC PEACE VIOLATION',
    'ROBBERY', 'SEX OFFENSE', 'STALKING', 'THEFT', 'WEAPONS VIOLATION', 'total']
    W = generate_transition_SocialLag(year)
#    i = retrieve_income_features()
#    e = retrieve_education_features()
#    r = retrieve_race_features()
    predCrimes = {}
    unpredCrimes = {}
    for idx, val in enumerate(header):
        Y = retrieve_crime_count(year, [val])
        
        f1 = np.dot(W, Y)
        f = np.concatenate( (f1, np.ones(f1.shape)), axis=1 )
        Y = Y.reshape((77,))
        
        # linearRegression(f1, Y)
        cnt = 0
        for z in Y:
            if z == 0:
                cnt += 1
        print ",".join( [val, str(cnt), ""] ),  # sparseness
        res = negativeBinomialRegression(f, Y)
        if res.mle_retvals['converged']:
            predCrimes[val] = [cnt, len(Y)]
        else:
            unpredCrimes[val] = [cnt, len(Y)]
        
    return predCrimes, unpredCrimes
    
    
    

def generate_flowType_crimeCount_matrix():
    """
    The result is used in ./plotMat.py/plot_flowType_crimeCount() function.
    
    Results shown on the wikispace:
    https://wikispaces.psu.edu/display/LSP/Social+flow%2C+Crime
    
    9/18/2015 Under which scenarios does the social lag help the most?
    """
    
    header = ['ARSON', 'ASSAULT', 'BATTERY', 'BURGLARY', 'CRIM SEXUAL ASSAULT', 
    'CRIMINAL DAMAGE', 'CRIMINAL TRESPASS', 'DECEPTIVE PRACTICE', 
    'GAMBLING', 'HOMICIDE', 'INTERFERENCE WITH PUBLIC OFFICER', 
    'INTIMIDATION', 'KIDNAPPING', 'LIQUOR LAW VIOLATION', 'MOTOR VEHICLE THEFT', 
    'NARCOTICS', 'OBSCENITY', 'OFFENSE INVOLVING CHILDREN', 'OTHER NARCOTIC VIOLATION',
    'OTHER OFFENSE', 'PROSTITUTION', 'PUBLIC INDECENCY', 'PUBLIC PEACE VIOLATION',
    'ROBBERY', 'SEX OFFENSE', 'STALKING', 'THEFT', 'WEAPONS VIOLATION', 'total']
    
    errors = np.zeros((9, len(header)))
    mre1 = np.zeros((9, len(header)))
    mre2 = np.zeros((9, len(header)))
    for idx, val in enumerate(header):
        for j in range(9):
            r1 = leaveOneOut_evaluation_onChicagoCrimeData(2010, ['corina'], crime_idx=idx+1, flow_type=j)
            r2 = leaveOneOut_evaluation_onChicagoCrimeData(2010, ['corina', 'sociallag'], crime_idx=idx+1, flow_type=j)
            mre1[j][idx] = r1[0,2]
            mre2[j][idx] = r2[0,2]
            errors[j][idx] = r1[0,2] - r2[0,2]
    np.savetxt('errors.array', errors)
    np.savetxt('mre1.array', mre1)
    np.savetxt('mre2.array', mre2)
    




def permutationTest_accuracy(iters, permute='taxiflow'):
    """
    Evaluate crime rate
    
    use full feature set:
        Corina, spaitallag, taxiflow, POIdist
    evaluate on 2013
    
    at CA level
    
    leave one out
    
    permutation
        permute one feature 1000 times takes roughly 30-40 minutes.
        The results are dumped as "permute-{feature}.pickle"
    """
    poi_dist = getFourSquarePOIDistribution(useRatio=False)
    F_taxi = getTaxiFlow(normalization="bydestination")
    W2 = generate_geographical_SpatialLag_ca()
    Y = retrieve_crime_count(year=2013)
    
    
    C = generate_corina_features()
    D = C[1]
    
    popul = C[1][:,0].reshape(C[1].shape[0],1)
    Y = np.divide(Y, popul) * 10000
    
     
    f2 = np.dot(W2, Y)
    ftaxi = np.dot(F_taxi, Y)
    
    
    nb_mae = []
    nb_mre = []
    lr_mae = []
    lr_mre = []
    for i in range(iters):
        if permute == 'corina':
            D = np.random.permutation(D)
        elif permute == 'spatiallag':
            yhat = np.random.permutation(Y)
            f2 = np.dot(W2, yhat)
        elif permute == 'taxiflow':            
            yhat = np.random.permutation(Y)
            ftaxi = np.dot(F_taxi, Y)
        elif permute == 'POIdist':
            poi_dist = np.random.permutation(poi_dist)
        f = np.ones(f2.shape)
        f = np.concatenate( (f, D, f2, ftaxi, poi_dist), axis=1 )
        header = ['intercept'] + C[0] + [ 'spatiallag', 'taxiflow'] + \
            ['POI food', 'POI residence', 'POI travel', 'POI arts entertainment', 
                           'POI outdoors recreation', 'POI education', 'POI nightlife', 
                           'POI professional', 'POI shops', 'POI event']
        df = pd.DataFrame(f, columns=header)
        
        np.savetxt("Y.csv", Y, delimiter=",")
        df.to_csv("f.csv", sep=",", index=False)
        
        # NB permute
        nbres = subprocess.check_output( ['Rscript', 'nbr_eval.R', 'ca'] )
        ls = nbres.split(' ')
        nb_mae.append( float(ls[0]) )
        nb_mre.append( float(ls[2]) )

        mae2, mre2 = permutation_Test_LR(Y, f)
        lr_mae.append(mae2)
        lr_mre.append(mre2)
        
        if i % 10 == 0:
            print i
        
    print '{0} iterations finished.'.format(iters)
    print pvalue(412.305, lr_mae), pvalue(0.363, lr_mre), \
        pvalue(319.86, nb_mae), pvalue(0.281, nb_mre)
    return nb_mae, nb_mre, lr_mae, lr_mre
    
    
def pvalue(v, l):
    cnt = 0.
    for e in l:
        if e < v:
            cnt += 1
    
    return cnt / len(l)


def permutation_Test_LR(Y, f):
    
    Y = Y.reshape((len(Y),))
    loo = cross_validation.LeaveOneOut(len(Y))
    
    errors = []
    for train_idx, test_idx in loo:
        f_train, f_test = f[train_idx], f[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        r = linearRegression(f_train, Y_train)
        y = r.predict(f_test)
        errors.append(np.abs(Y_test - y))
        
    mae = np.mean(errors)
    mre = mae / Y.mean()
    
    return mae, mre



from sklearn.preprocessing import MinMaxScaler


def NB_coefficients(year=2010):
    poi_dist = getFourSquarePOIDistribution(useRatio=False)
    F_taxi = getTaxiFlow(normalization="bydestination")
    W2 = generate_geographical_SpatialLag_ca()
    Y = retrieve_crime_count(year=year)
    C = generate_corina_features()
    D = C[1]

    popul = C[1][:,0].reshape(C[1].shape[0],1)
    Y = np.divide(Y, popul) * 10000
    
    f2 = np.dot(W2, Y)
    ftaxi = np.dot(F_taxi, Y)
    
    f = np.concatenate( (D, f2, ftaxi, poi_dist), axis=1 )
    mms = MinMaxScaler(copy=False)
    mms.fit(f)
    mms.transform(f)
    header = C[0] + [ 'spatiallag', 'taxiflow'] + \
        ['POI food', 'POI residence', 'POI travel', 'POI arts entertainment', 
                       'POI outdoors recreation', 'POI education', 'POI nightlife', 
                       'POI professional', 'POI shops', 'POI event']
    df = pd.DataFrame(f, columns=header)
    
    np.savetxt("Y.csv", Y, delimiter=",")
    df.to_csv("f.csv", sep=",", index=False)
    
    # NB permute
    nbres = subprocess.check_output( ['Rscript', 'nbr_eval.R', 'ca', 'coefficient'] )
    print nbres
    
    ls = nbres.strip().split(" ")
    coef = [float(e) for e in ls]
    print coef
    return coef, header



def coefficients_pvalue(lehdType="total", crimeType='total'):
    """Return the pvalue of Negative Binomial model coefficients.
    Permutation test + leave-one-out evaluation
    Retrieve leave-one-out error distribution. To determine the p-value
    
    The model to be evaluated is the NB model.
    The features used in this model only includes spatial lag, scial lag, and demographics.
    
    Keyword arguments:
    lehdType -- the type of LEHD flow (default "total", alternative "lowincome")
    crimeType -- the type of predicated crime (default "violent", alternative "total")
    """
    
    C = generate_corina_features('ca')
    demo = pd.DataFrame(data=C[1], columns=C[0], dtype="float")
    W1 = generate_geographical_SpatialLag_ca()
    
    # the LEHD type
    if lehdType == "lowincome":
        W2 = generate_transition_SocialLag(year=2010, lehd_type=4, region='ca',
                                       normalization='none')
    elif lehdType == "total":
        W2 = generate_transition_SocialLag(year=2010, lehd_type=0, region='ca',
                                           normalization='none')
    elif lehdType == "taxi":
        W2 = getTaxiFlow(normalization="none")
                                           
    
    # the predicated crime type                                           
    violentCrime = ['HOMICIDE', 'CRIM SEXUAL ASSAULT', 'BATTERY', 'ROBBERY', 
                'ARSON', 'DOMESTIC VIOLENCE', 'ASSAULT']
    if crimeType == 'total':
        Y = retrieve_crime_count(year=2010, col=['total'], region='ca')
    elif crimeType == 'violent':
        Y = retrieve_crime_count(year=2010, col=violentCrime, region='ca')
    
        
    demo.to_csv("../R/pvalue-demo.csv", index=False)
    np.savetxt("../R/pvalue-spatiallag.csv", W1, delimiter=",")
    np.savetxt("../R/pvalue-sociallag.csv", W2, delimiter=",")
    np.savetxt("../R/pvalue-crime.csv", Y)
    

    # use a multiprocess Pool to run subprocess in parallel
    socialNorm = ['bydestination', 'bysource', 'bypair']
    os.chdir("../R")
    from multiprocessing import Pool, cpu_count
    subProcessPool = Pool(cpu_count() / 2)
    itersN = "1000"

    for sociFlag in ["useLEHD", "noLEHD"][0:1]:
        for geoFlag in ["useGeo", "noGeo"][0:1]:
            for sn in socialNorm[0:1]:
                for ep in ["exposure", "noexposure"][0:1]:
                    for logpop in ["logpop", "pop"][0:1]:
                        for logpopden in ["logpopdensty", "popdensty"][0:1]:
                            subProcessPool.apply_async(subPworker, (lehdType, crimeType, sn, ep, logpop, sociFlag, geoFlag, itersN, logpopden))
                            #subprocess.Popen(['Rscript', 'pvalue-evaluation.R', lehdType+"lehd", crimeType+"crime", sn, ep, logpop])
    subProcessPool.close()
    subProcessPool.join()
    


def subPworker(lehdType, crimeType, sn, ep, logpop, sociFlag, geoFlag, itersN, logpopden):
    print "Start worker with", sn, ep, logpop, sociFlag, geoFlag, itersN, logpopden
    import platform
    if platform.system() == "Windows":
        p = subprocess.Popen(['Rscript', 'pvalue-evaluation.R', lehdType+"lehd", crimeType+"crime", 
                          sn, ep, logpop, sociFlag, geoFlag, itersN, logpopden], shell=True)
    elif platform.system() == "Linux":
        p = subprocess.Popen(['Rscript', 'pvalue-evaluation.R', lehdType+"lehd", crimeType+"crime", 
                          sn, ep, logpop, sociFlag, geoFlag, itersN, logpopden])
        
    print p.pid, "is running"
    p.wait()
        



def longTable_features_allYears():
    """
    To serve the needs of Corina run her own experiemtns.
    Create the following tables in sperate tables containing all different years.
    """
    
    # Crime for different years
    violentCrime = ['HOMICIDE', 'CRIM SEXUAL ASSAULT', 'BATTERY', 'ROBBERY', 
                'ARSON', 'DOMESTIC VIOLENCE', 'ASSAULT']
    total_crime = []
    violent_crime = []
    homicide_crime = []
    
    # social lag
    norms = ['source', 'destination', 'pair']
    sociallag_low = dict( [(key, []) for key in norms])
    sociallag_all = dict( [(key, []) for key in norms])
    
    for year in range(2001, 2016):
        y = retrieve_crime_count(year, col=['total'], region='ca')
        total_crime.append(y)
        yv = retrieve_crime_count(year, col=violentCrime, region='ca')
        violent_crime.append(yv)
        yh = retrieve_crime_count(year, col=['HOMICIDE'], region='ca')
        homicide_crime.append(yh)
        
        if year >= 2002 and year <= 2013:
            for n in norms:
                # social lag with low income flow
                socialLOW = generate_transition_SocialLag(year=year, lehd_type=4, 
                                                          region='ca', normalization=n)
                                                          
                s_low = np.dot(socialLOW, y)
                sociallag_low[n].append(s_low)
                                                  
                socialALL = generate_transition_SocialLag(year=year, lehd_type=0,
                                                          region='ca', normalization=n)
                s_all = np.dot(socialALL, y)
                sociallag_all[n].append(s_all)
            
            
        
    crime_header = ','.join( [str(i) for i in  range(2001, 2016)] )
    
    total_crime = np.transpose(np.squeeze( np.array(total_crime) ))
    np.savetxt("total_crime.csv", total_crime, delimiter=",", fmt="%d", 
               header=crime_header, comments='')
               
    violent_crime = np.transpose(np.squeeze(np.array(violent_crime)))
    np.savetxt("violent_crime.csv", violent_crime, delimiter=",", fmt="%d",
               header=crime_header, comments='')
               
    homicide_crime = np.transpose(np.squeeze(np.array(homicide_crime)))
    np.savetxt("homicide_crime.csv", homicide_crime, delimiter=",", fmt="%d",
               header=crime_header, comments='')
            
    lag_header = ",".join( [str(i) for i in range(2002, 2014)])
    
    
    for n in norms:
        sl = np.transpose(np.squeeze(np.array(sociallag_low[n])))
        np.savetxt("sociallag_low_{0}.csv".format(n), sl, delimiter=",", fmt="%f",
                   header=lag_header, comments='')
                   
        
        sa = np.transpose(np.squeeze(np.array(sociallag_all[n])))
        np.savetxt("sociallag_all_{0}.csv".format(n), sa, delimiter=",", fmt="%f",
               header=lag_header, comments='')
    
    
    

if __name__ == '__main__':
    import sys
    t = sys.argv[1]
    print sys.argv
#   crimeRegression_eachCategory()
    # f = unitTest_onChicagoCrimeData()
#   print f.summary()
    if t == 'leaveOneOut':
        r = leaveOneOut_evaluation_onChicagoCrimeData(2010, features=
                 ['corina', 'spatiallag', 'sociallag', 'taxiflow', 'POIdist'],   # temporallag
                 verboseoutput=False, region='ca', logFeatures=['spatiallag2', 'sociallag2', 'taxiflow2'])
    elif t == 'permutation':
        permutationTest_onChicagoCrimeData(2010, ['corina', 'sociallag', 'spatiallag', 'temporallag'], iters=3)
    elif t == 'socialflow':
        for year in range(2002, 2014):
            W = generate_transition_SocialLag(year, lehd_type=0)
            np.savetxt(here + "/W-{0}.csv".format(year), W, delimiter="," )
    elif t == 'permuteAccu':
        r = permutationTest_accuracy(1000)
    elif t == 'coefficient':
        v = []
        for year in range(2010, 2015):
            r, h = NB_coefficients(year)
            v.append(r)
        v = np.array(v)
        visualize = False
        if visualize == True:
            plt.figure()        
            plt.plot(v[:,0], )
            plt.xticks(range(5), ('2010', '2011', '2012', '2013', '2014'))
            
            plt.figure()
            plt.plot(v[:,1:])
            plt.xticks(range(5), ('2010', '2011', '2012', '2013', '2014'))
        else:
            v0 = v[0]
            o = np.flipud(np.argsort(v0))
            for i in range(1,4):
                j = o[i]
                print ' &'.join( [h[j]] + ['{0:.3f}'.format(row[j]) for row in v] )
            for i in range(3, 0, -1):
                j = o[-i]
                print ' &'.join( [h[j]] + ['{0:.3f}'.format(row[j]) for row in v] )
    elif t == 'pvalue':
        coefficients_pvalue(lehdType="total", crimeType="total")
    elif t == 'getlongtable':
        r = longTable_features_allYears()
    
#    CV = '10Fold'
#    feat_candi = ['corina', 'spatiallag', 'temporallag', 'sociallag']
#    for i in range(1,5):
#        f_lists = combinations(feat_candi, i)
#        for f in f_lists:
#            print '+'.join(f),
#            if CV == '10Fold':
#                r = tenFoldCV_onChicagoCrimeData(f)
#            else:
#                r = tenFoldCV_onChicagoCrimeData(f, CVmethod='leaveOneOut')
    
    
    # Ps = range(1, 5) + range(10, 81, 20)
    # for p in Ps:
        # print p,
        # s1, s2 = tenFoldCV_onChicagoCrimeData(['temporallag'], CVmethod='leavePOut', P=p, NUM_ITER=20)
    
#    for num_iter in range(10, 41, 5):
#        print num_iter,
#        s1,s2 = tenFoldCV_onChicagoCrimeData(['temporallag'], CVmethod='leaveOneOut', NUM_ITER=20, SHUFFLE=False)
