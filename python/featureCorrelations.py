# -*- coding: utf-8 -*-
"""
Feature correlation analysis



Created on Sun Jan 31 21:09:18 2016

@author: kok
"""


import numpy as np
from FeatureUtils import retrieve_crime_count
from Crime import Tract
from foursquarePOI import getFourSquarePOIDistribution




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
    


from FeatureUtils import generate_transition_SocialLag


def correlation_socialflow_crime():
    """
    calculate the correlation between social flow and crime count.
    
    
if __name__ == '__main__':
    
   correlation_POIdist_crime()