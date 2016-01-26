# -*- coding: utf-8 -*-
"""
Generate the taxi flow in Chicago.


Created on Mon Jan 25 16:19:18 2016

@author: kok
"""


from Crime import Tract
from shapely.geometry import Point
import csv
import numpy as np

import os.path

here = os.path.dirname(os.path.abspath(__file__))





def getTaxiFlow(leaveOut = -1):
    """
    Retrieve taxi flow from file
    
    the self-flow is set to zero.
    
    LeaveOut define the region to be left out for evaluation. Value 
    ranges from 1 to 77
    """
    s = np.loadtxt("TF.csv", delimiter=",")
    n = s.shape[0]
    for i in range(n):
        s[i,i] = 0
    
    if leaveOut > 0:
        s = np.delete(s, leaveOut -1, 0)
        s = np.delete(s, leaveOut -1, 1)
    return s


if __name__ == '__main__':
    
    cas = Tract.createAllCAObjects()
    n = len(cas)
    TF = np.zeros((n, n))   # taxi flow matrix
    
#    cnt = 0
    
    with open('../data/ChicagoTaxi/201401-03.txt', 'rU') as fin:
        reader = csv.reader(fin, delimiter='\t' )
        header = reader.next()
        for row in reader:
            # initialize points            
            start = Point(float(row[3]), float(row[4]))
            end = Point(float(row[5]), float(row[6]))
            
            sid = -1
            eid = -1
            for key, grid in cas.items():
                """
                grid key starts from 1
                map the start/end point of trip into grids to get flow
                """
                if grid.polygon.contains(start):
                    sid = key - 1
                if grid.polygon.contains(end):
                    eid = key - 1
                if sid != -1 and eid != -1:
                    break
            
            TF[sid, eid] += 1
#            cnt += 1
#            if (cnt > 1000):
#                break

    np.savetxt(here + "/TF.csv", TF, delimiter="," )