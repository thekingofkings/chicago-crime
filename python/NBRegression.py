"""
Run the Negative Binomial Regression model

The regresion model use various features to predict the crime count in each
unit of study area (i.e. tract or Community Area)

Author: Hongjian
date:8/20/2015
"""

from Crime import Tract
from sets import Set
from shapely.geometry import MultiLineString
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from statsmodels.base.model import GenericLikelihoodModel


"""
Part One
Generate vairous features
"""


def generate_geographical_SocialLag(foutName):
    """
    Generate the spatial lag from the geographically adjacent CAs.
    """
    fout = open(foutName, 'w')
    cnt = 0
    ts = Tract.createAllTractObjects()
    idset = Set(ts.keys())
    for i in ts.keys():
        idset.remove(i)
        for j in idset:
            if type(ts[i].polygon.intersection(ts[j].polygon)) is MultiLineString:
                fout.write('{0},{1}\n'.format(i,j))
                cnt += 1
    fout.close()
    return cnt
        
        



def generate_transition_SocialLag(year = 2010):
    """
    Generate the spatial lag from the transition flow connected CAs.
    """
    listIdx = {}
    fin = open('../data/chicago_ca_od_{0}.csv'.format(year))
    for line in fin:
        ls = line.split(",")
        srcid = int(ls[0])
        dstid = int(ls[1])
        val = int(ls[2])
        if srcid in listIdx:
            listIdx[srcid][dstid] = val
        else:
            listIdx[srcid] = {}
            listIdx[srcid][dstid] = val                            
    fin.close()

    W = np.zeros( (77,77) )
    for srcid, sdict in listIdx.items():
        total = (float) (sum( sdict.values() ))
        for dstid, val in sdict.items():
            if srcid != dstid:
                W[srcid-1][dstid-1] = val / total

    return W




def retrieve_crime_count(year, col=-1):
    """
    Retrieve the crime count in a vector
    Input:
        year - the year to retrieve
        col  - the type of crime
    """
    Y =np.zeros( (77,1) )
    with open('../data/chicago-crime-ca-level-{0}.csv'.format(year)) as fin:
        for line in fin:
            ls = line.split(",")
            idx = int(ls[0])
            val = int(ls[col])
            Y[idx-1] = val

    return Y

    

def linearRegression(features, Y):
    """
    learn the linear regression model from features to Y
    output the regression analysis parameters
    plot scatter plot
    """
    sl, intcpt, rval, pval, stderr = stats.linregress(f1, Y)
    print sl, intcpt, rval, pval, stderr
    plt.scatter(f1, Y)


    
    

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
        ll = nbinom.logpmf( self.endog, size, prob)
        return - ll
        
    def fit(self, start_params=None, maxiter = 10000, maxfun=10000, **kwds):
        if start_params == None:
            start_params = np.append(np.zeros(self.exog.shape[1]), .5)
            start_params[0] = np.log(self.endog.mean())
        return super(NegBin, self).fit(start_params=start_params, maxiter=maxiter,
                maxfun=maxfun, **kwds)
                
        
  

    
def negativeBinomialRegression(features, Y):
    """
    learn the NB regression
    """
    mod = NegBin(Y, features)
    res = mod.fit()
    if res.mle_retvals['converged']:
        print res.summary()
    return res


    
def unitTest_negativeBinomialRegression():
    import patsy
    import pandas as pd
    url = 'http://vincentarelbundock.github.com/Rdatasets/csv/COUNT/medpar.csv'
    medpar = pd.read_csv(url)
    y, X = patsy.dmatrices('los~type2+type3+hmo+white', medpar)
    
    
if __name__ == '__main__':
    # generate_geographical_SocialLag('../data/chicago-CA-geo-neighbor')
    header = ['ARSON', 'ASSAULT', 'BATTERY', 'BURGLARY', 'CRIM SEXUAL ASSAULT', 
    'CRIMINAL DAMAGE', 'CRIMINAL TRESPASS', 'DECEPTIVE PRACTICE', 
    'GAMBLING', 'HOMICIDE', 'INTERFERENCE WITH PUBLIC OFFICER', 
    'INTIMIDATION', 'KIDNAPPING', 'LIQUOR LAW VIOLATION', 'MOTOR VEHICLE THEFT', 
    'NARCOTICS', 'OBSCENITY', 'OFFENSE INVOLVING CHILDREN', 'OTHER NARCOTIC VIOLATION',
    'OTHER OFFENSE', 'PROSTITUTION', 'PUBLIC INDECENCY', 'PUBLIC PEACE VIOLATION',
    'ROBBERY', 'SEX OFFENSE', 'STALKING', 'THEFT', 'WEAPONS VIOLATION', 'total']
    W = generate_transition_SocialLag(2010)
    for idx, val in enumerate(header):
        Y = retrieve_crime_count(2010, idx+1)
        
        f1 = np.dot(W, Y).reshape((77,))
        Y = Y.reshape((77,))
        
        # linearRegression(f1, Y)
        print val
        res = negativeBinomialRegression(f1, Y)
