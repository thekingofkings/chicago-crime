# -*- coding: utf-8 -*-
"""
Generate the foursquare POI feature of Chicago.
Use the POI data at
    ../data/all_POIs_chicago

Created on Tue Jan 26 11:09:49 2016

@author: kok
"""


from Crime import Tract
from shapely.geometry import Point
import pickle
import numpy as np
        
import os.path

here = os.path.dirname(os.path.abspath(__file__)) 




def getFourSquareCount(leaveOut = -1):
    """
    retrieve Foursquare count from local file
    """
    s = np.loadtxt("POI_cnt.csv", delimiter=",")
    
    if leaveOut > 0:
        s = np.delete(s, leaveOut-1, 0)
    return s
    

def getFourSquarePOIDistributionHeader():
    header = ['Food', 'Residence', 'Travel', 'Arts & Entertainment', 
        'Outdoors & Recreation', 'College & Education', 'Nightlife', 
        'Professional', 'Shops', 'Event']
    return header
    
    
def getFourSquarePOIDistribution( leaveOut = -1, gridLevel = 'ca', useRatio=False):
    """
    retrieve Foursquare POI distribution from local file
    """
    if gridLevel == 'ca':
        d = np.loadtxt("POI_dist.csv", delimiter=",")
    elif gridLevel == 'tract':
        d = np.loadtxt('POI_dist_tract.csv', delimiter=",")
    
    if leaveOut > 0:
        d = np.delete(d, leaveOut-1, 0)
        
    if useRatio:
        poi_sum = np.sum(d, axis=1)
        for i in range(len(poi_sum)):
            d[i] = d[i] / poi_sum[i]
        d = np.nan_to_num(d)
    
    return d
    
    
    
def generatePOIfeature(gridLevel = 'ca'):
    """
    generate POI features and write out to a file
    
    regionLevel could be "ca" or "tract"
    
    ['Food', 'Residence', 'Travel', 'Arts & Entertainment', 
    'Outdoors & Recreation', 'College & Education', 'Nightlife', 
    'Professional', 'Shops', 'Event']
    """
    if gridLevel == 'ca':
        cas = Tract.createAllCAObjects()
    elif gridLevel == 'tract':
        cas = Tract.createAllTractObjects()

    ordKey = sorted(cas.keys())
    
    gcn = np.zeros((len(cas), 3))   # check-in count, user count, and POI count
    gcat = {}
    
    with open('../data/all_POIs_chicago', 'r') as fin:
        POIs = pickle.load(fin)
        
    with open('category_hierarchy.pickle', 'r') as f2:
        poi_cat = pickle.load(f2)
    
    cnt = 0
    for poi in POIs.values():
        loc = Point(poi.location.lon, poi.location.lat)
        if poi.cat in poi_cat:
            cat = poi_cat[poi.cat]
        else:
            continue
        
        for key, grid in cas.items():
            if grid.polygon.contains(loc):
                gcn[ordKey.index(key),0] += poi.checkin_count
                gcn[ordKey.index(key),1] += poi.user_count
                gcn[ordKey.index(key),2] += 1
                """
                Build a two-level dictionary,
                first index by region id,
                then index by category id,
                finally, the value is number of POI under the category.
                """
                if key in gcat:
                    if cat in gcat[key]:
                        gcat[key][cat] += 1
                    else:
                        gcat[key][cat] = 1
                else:
                    gcat[key] = {}
                    gcat[key][cat] = 1
                    
                # break the polygon loop
                cnt += 1
                break
    
    s = 0
    hi_catgy = []
    for catdict in gcat.values():
        hi_catgy += catdict.keys()
        for c in catdict.values():
            s += c
            
    hi_catgy = list(set(hi_catgy))
    print hi_catgy
    
    
    gdist = np.zeros( (len(cas), len(hi_catgy)) )
    for key, distDict in gcat.items():
        for idx, cate in enumerate(hi_catgy):
            if cate in distDict:            
                gdist[ordKey.index(key), idx] = distDict[cate]
            else:
                gdist[ordKey.index(key), idx] = 0
                
    if gridLevel == 'ca':
        np.savetxt(here + "/POI_dist.csv", gdist, delimiter="," )
        np.savetxt(here + "/POI_cnt.csv", gcn, delimiter="," )
    elif gridLevel == 'tract':
        np.savetxt(here + "/POI_dist_tract.csv", gdist, delimiter="," )
        np.savetxt(here + "/POI_cnt_tract.csv", gcn, delimiter="," )
    
    


    

if __name__ == '__main__':
    
    generatePOIfeature(gridLevel='ca')
#    d = getFourSquarePOIDistribution(useRatio=True)
#    np.savetxt("../R/poi_dist.csv", d, delimiter=",")