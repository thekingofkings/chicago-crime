# -*- coding: utf-8 -*-
"""
Feature correlation analysis



Created on Sun Jan 31 21:09:18 2016

@author: kok
"""


import numpy as np
from FeatureUtils import *
from Crime import Tract
from foursquarePOI import getFourSquarePOIDistribution, getFourSquarePOIDistributionHeader
from scipy.stats import pearsonr

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = (4*1.6,3*1.7)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42




def correlation_POIdist_crime():
    """
    we calculate the correlation between POI distribution and crime for each
    community area(CA).
    Within each CA, the crime count is number of crime in each tract.
    The POI count is number of POIs in each tract.
    """
    tracts = Tract.createAllTractObjects()
    ordkey = sorted(tracts.keys())
    CAs = {}
    for key, val in tracts.items():
        if val.CA not in CAs:
            CAs[val.CA] = [key]
        else:
            CAs[val.CA].append(key)
    
    Y = retrieve_crime_count(2010, col=['total'], region='tract')
    poi_dist = getFourSquarePOIDistribution(gridLevel='tract')
    
    
    Pearson = {}
    for cakey, calist in CAs.items():
        crime = []
        pois = []
        for tractkey in calist:
            crime.append(Y[tractkey])
            pois.append(poi_dist[ordkey.index(tractkey)])
        # calculate correlation
        pois = np.array(pois)
        crime = np.array(crime)
        pearson = []
        for i in range(pois.shape[1]):
            r = np.vstack( (pois[:,i], crime) )
            pearson.append( np.corrcoef(r)[0,1] )
            
        Pearson[cakey] = np.nan_to_num( pearson )

    P = []
    for key in range(1, 78):
        P.append(Pearson[key])
    
    np.savetxt("../R/poi_correlation_ca.csv", P, delimiter=",")
    return np.array(P)
    




def correlation_POI_crime(gridLevel='tract', poiRatio=False):
    """
    calculate correlation for different POI category
    """
    Y = retrieve_crime_count(2010, col=['total'], region=gridLevel)
    h, D = generate_corina_features(region='ca')
    popul = D[:,0].reshape(D.shape[0],1)
    poi_dist = getFourSquarePOIDistribution(gridLevel=gridLevel, useRatio=poiRatio)
    cate_label = ['Food', 'Residence', 'Travel', 'Arts & Entertainment', 
                'Outdoors & Recreation', 'College & Education', 'Nightlife', 
                'Professional', 'Shops', 'Event']
    
    if gridLevel == 'tract':
        tracts = Tract.createAllTractObjects()
        ordkey = sorted(tracts.keys())

        crime = []
        pois = []
        for tractkey in ordkey:
            crime.append(Y[tractkey])
            pois.append(poi_dist[ordkey.index(tractkey)])
        
        pois = np.array(pois)
        crime = np.array(crime)
    
        for i in range(pois.shape[1]):
            r = np.vstack( (pois[:,i], crime) )
            pcc = np.corrcoef(r)[0,1]
            print pcc
            
    elif gridLevel == 'ca':
        Y = np.divide(Y, popul) * 10000
        Y = Y.reshape( (len(Y),) )
        poi_dist = np.transpose(poi_dist)
        
        for i in range(poi_dist.shape[0]):
            poi = np.reshape(poi_dist[i,:], Y.shape )
            r, p = pearsonr(poi, Y)
            print cate_label[i], r, p



def line_POI_crime():
    d = getFourSquarePOIDistribution(gridLevel='ca')
    y = retrieve_crime_count(2010, col=['total'], region='ca')
    h, D = generate_corina_features(region='ca')
    popul = D[:,0].reshape(D.shape[0],1)
    
    hd = getFourSquarePOIDistributionHeader()
    yhat = np.divide(y, popul) * 10000
    
    for i in range(6,8):
        plt.figure()
        plt.scatter(d[:,i], y)
        plt.xlim(0, 1000)
        plt.xlabel('POI count -- {0} category'.format(hd[i]))
        plt.ylabel('Crime count')
        
    
        
        plt.figure()
        plt.scatter(d[:,i], yhat)
        plt.xlim(0, 1000)
        plt.xlabel('POI count -- {0} category'.format(hd[i]))
        plt.ylabel('Crime rate (per 10,000)')





def correlation_demo_crime():
    """
    demographics correlation with crime
    """
    Y = retrieve_crime_count(year=2010, col=['total'], region='ca')
    h, D = generate_corina_features(region='ca')
    print h
    popul = D[:,0].reshape(D.shape[0],1)
    Y = np.divide(Y, popul) * 10000
    
    Y = Y.reshape( (len(Y),) )
    D = D.transpose()
    for i in range(D.shape[0]):
        demo = D[i,:].reshape( (Y.shape ) )
        r, p = pearsonr(demo, Y)
        print r, p
    
    





def correlation_socialflow_crime(region='tract', useRate=False, 
                                 weightSocialFlow=False):
    """
    calculate the correlation between social flow and crime count.
    """
    if region == 'ca':
        W = generate_transition_SocialLag(region='ca')
        W2 = generate_geographical_SpatialLag_ca()
        Y = retrieve_crime_count(2010, region='ca')
    elif region == 'tract':
        W = generate_transition_SocialLag(region='tract')
        W2, tractkey = generate_geographical_SpatialLag()
        Ymap = retrieve_crime_count(2010, col=['total'], region='tract')
        Y = np.array( [Ymap[k] for k in tractkey] ).reshape(len(Ymap), 1)
    
    
    U = generate_corina_features(region)
    if useRate:
        print 'Use crime rate per 10,000 population'
        
        if region == 'tract':
            C_mtx = []
            cnt = 0
            
            for k in tractkey:
                if k in U[1]:
                    C_mtx.append(U[1][k])
                else:
                    cnt += 1
                    C_mtx.append( [1] + [0 for i in range(6)] ) # population 1
        
            U = ( U[0], np.array( C_mtx ) )
            print len(tractkey), cnt
            
        popul = U[1][:,0].reshape(U[1].shape[0],1)
        Y = np.divide(Y, popul) * 10000
        
        
    if weightSocialFlow:
        wC = 130.0 if useRate else 32.0     # constant parameter
        poverty = U[1][:,2]        
        for i in range(W.shape[0]):
            for j in range (W.shape[1]):
                s = np.exp( - np.abs(poverty[i] - poverty[j]) / wC )
                W[i][j] *= s
    
    f1 = np.dot(W, Y)    
    r = np.transpose( np.hstack( (Y, f1) ) )
    pcc1 = np.corrcoef(r)
    
    
    f2 = np.dot(W2, Y)
    r = np.transpose( np.hstack( (Y, f2) ) )
    pcc2 = np.corrcoef(r)
    
    print '{0}: social lag {1}, spatial lag {2}'.format(region, pcc1[0,1], pcc2[0,1])




def line_socialflow_crime():
    W = generate_transition_SocialLag(region='ca')
    
    C = generate_corina_features()
    poverty = C[1][:,2]        
    for i in range(W.shape[0]):
        for j in range (W.shape[1]):
            W[i][j] *= np.exp( - np.abs(poverty[i] - poverty[j]) / 32 )
            
            
    Y = retrieve_crime_count(2010, col=['total'], region='ca')
    h, D = generate_corina_features(region='ca')
    popul = D[:,0].reshape(D.shape[0],1)
    Y = np.divide(Y, popul) * 10000
    f1 = np.dot(W, Y)
    
    
    plt.scatter(f1, Y)
    plt.xlabel('Social lag weighted by demographic similarity')
    plt.ylabel('crime rate')



    
    
    

def line_spatialflow_crime():
    W = generate_geographical_SpatialLag_ca()
            
    Y = retrieve_crime_count(2010, col=['total'], region='ca')
    h, D = generate_corina_features(region='ca')
    popul = D[:,0].reshape(D.shape[0],1)
    Y = np.divide(Y, popul) * 10000
    f1 = np.dot(W, Y)
    
    plt.figure()
    plt.scatter(f1, Y)
    plt.axis([0,700000, 0, 6000])
    idx = [31, 75, 37]
    sf1 = f1[idx]
    sY = Y[idx]
    plt.scatter(sf1, sY, edgecolors='red', s=50, linewidths=2 )
    plt.figtext(0.43, 0.78, '#32', fontsize='large')
    plt.figtext(0.15, 0.37, '#76', fontsize='large')    
    plt.figtext(0.79, 0.33, '#38', fontsize='large')
    plt.xlabel('Geographical influence feature value', fontsize='x-large')
    plt.ylabel('Crime rate', fontsize='x-large')
    
    plt.savefig('spatial-crime-rate.pdf', format='pdf')
    return Y
    
    
    
from taxiFlow import getTaxiFlow

def line_taxiflow_crime():
    s = getTaxiFlow(normalization='bydestination')
    
    Y = retrieve_crime_count(2010, col=['total'], region='ca')
    h, D = generate_corina_features(region='ca')
    popul = D[:,0].reshape(D.shape[0],1)
    Y = np.divide(Y, popul) * 10000
    
    f1 = np.dot(s, Y)
    
    plt.figure()
    plt.scatter(f1, Y)
    plt.axis([0, 6000, 0, 6000])
    idx = [31, 46]
    sf1 = f1[idx]
    sY = Y[idx]
    plt.scatter(sf1, sY, edgecolors='red', s=50, linewidths=2 )
    plt.figtext(0.33, 0.8, '#32', fontsize='large')
    plt.figtext(0.75, 0.34, '#47', fontsize='large')
    plt.xlabel('Hyperlink by taxi flow feature value', fontsize='x-large')
    plt.ylabel('Crime rate', fontsize='x-large')
    
    plt.savefig('taxi-flow-percent.pdf', format='pdf')
    return f1
    






def correlation_taxiflow_crime(flowPercentage=True, crimeRate=True):
    """
    correlation between taxi flow and crime
    """
    s = getTaxiFlow(usePercentage=flowPercentage)
    Y = retrieve_crime_count(2010, region='ca')
    if crimeRate:
        h, D = generate_corina_features(region='ca')
        popul = D[:,0].reshape(D.shape[0],1)
        Y = np.divide(Y, popul) * 10000
    
    f1 = np.dot(s, Y)
    r = np.hstack( (f1, Y) )
    r = np.transpose(r)
    pcc = np.corrcoef(r)
    print pcc




if __name__ == '__main__':
    
#    correlation_POIdist_crime()
#    correlation_POI_crime('ca')
#    r = line_taxiflow_crime()
#    line_POI_crime()
#    line_socialflow_crime()
#    r = line_spatialflow_crime()
#    correlation_socialflow_crime(region='ca', useRate=True, weightSocialFlow=True)
    r = correlation_demo_crime()
#    correlation_taxiflow_crime(flowPercentage=True, crimeRate=True)
