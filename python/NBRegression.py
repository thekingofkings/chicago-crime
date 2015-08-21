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



"""
Part One
Generate vairous features
"""


def generate_geographical_SpatialLag(foutName):
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
        
        



def generate_transition_SpatialLag():
    """
    Generate the spatial lag from the transition flow connected CAs.
    """
    listIdx = {}
    fin = open('../data/chicago_ca_od_2010.csv')
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




def retrieve_crime_count():
    """
    Retrieve the crime count in a vector
    """
    Y =np.zeros( (77,1) )
    with open('../data/chicago-crime-ca-level-2010.csv') as fin:
        for line in fin:
            ls = line.split(",")
            idx = int(ls[0])
            val = int(ls[1])
            Y[idx-1] = val

    return Y



if __name__ == '__main__':
    # generate_geographical_SpatialLag('../data/chicago-CA-geo-neighbor')
    W = generate_transition_SpatialLag()
    Y = retrieve_crime_count()
    
    f1 = np.dot(W, Y).reshape((77,))
    Y = Y.reshape((77,))
    sl, intcpt, rval, pval, stderr = stats.linregress(f1, Y)
    print sl, intcpt, rval, pval, stderr
    plt.scatter(f1, Y)
