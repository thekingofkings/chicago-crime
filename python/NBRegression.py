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
    




if __name__ == '__main__':
    generate_geographical_SpatialLag('../data/chicago-CA-geo-neighbor')
