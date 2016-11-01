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





def getTaxiFlow(leaveOut = -1, normalization="bydestination", gridLevel='ca'):
    """
    Retrieve taxi flow from file
    
    the self-flow is set to zero.
    
    LeaveOut define the region to be left out for evaluation. Value 
    ranges from 1 to 77
    
    normalization takes value "none/bydestination/bysource"
    """
    if gridLevel == 'ca':
        s = np.loadtxt("TF.csv", delimiter=",")
    elif gridLevel == 'tract':
        s = np.loadtxt("TF_tract.csv", delimiter=",")
    n = s.shape[0]
    
    for i in range(n):
        s[i,i] = 0
    
    if leaveOut > 0:
        s = np.delete(s, leaveOut -1, 0)
        s = np.delete(s, leaveOut -1, 1)
        
    n = s.shape[0]
    
    return taxi_flow_normalization(s, normalization)
    


def taxi_flow_normalization(tf, method="bydestination"):
    """
    Normalize the taxi flow matrix `tf`.
    Input:
    tf - raw taxi flow matrix tf_ij is flow from i to j
    method - choice of normalization
    
    Output:
    tf_norm - normlized taxi flow matrix, tf_norm_ij is flow from j to i
    """
    n = tf.shape[0]
    tf = tf.astype(float)
    try:
        assert tf.dtype == "float64"
        if method == "bydestination":
            tf = np.transpose(tf)
            fsum = np.sum(tf, axis=1, keepdims=True)
            fsum[fsum==0] = 1
            assert fsum.shape == (n,1)
            tf = tf / fsum
            assert tf.sum() == n
            np.testing.assert_almost_equal(tf.sum(axis=1)[n-1], 1)
        elif method == "bysource":
            fsum = np.sum(tf, axis=1, keepdims=True)
            fsum[fsum==0] = 1
            tf = tf / fsum
            assert fsum.shape == (n,1)
            np.testing.assert_almost_equal(tf.sum(axis=1)[n-1], 1)
        elif method == "none":
            pass
        else:
            print "Normalization method is not implemented."
    except AssertionError as err:
        print tf.sum(), n, err
        
    return tf
            



def generateTaxiFlow(gridLevel='ca'):
    """
    Generate taxi flow and write it to a file
    
    This is slow to run
    """
    if gridLevel == 'ca':
        cas = Tract.createAllCAObjects()
    elif gridLevel == 'tract':
        cas = Tract.createAllTractObjects()
    n = len(cas)
    TF = np.zeros((n, n))   # taxi flow matrix
    
    ordKey = sorted(cas.keys())
    
    cnt = 0
#    import os
#    fnames = os.listdir("../data/ChicagoTaxi/")
    fnames = ['201401-03.txt']
    
    for fname in fnames:
        print "Count taxi flow in {0}".format(fname)
        with open('../data/ChicagoTaxi/{0}'.format(fname), 'rU') as fin:
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
                        sid = ordKey.index(key)
                    if grid.polygon.contains(end):
                        eid = ordKey.index(key)
                    if sid != -1 and eid != -1:
                        break
                
                TF[sid, eid] += 1
                cnt += 1
                if (cnt % 100000 == 0):
                    print "{0} trips have been added".format(cnt)
    if gridLevel == 'ca':
        np.savetxt(here + "/TF.csv", TF, delimiter="," )
    elif gridLevel == 'tract':
        np.savetxt(here + "/TF_tract.csv", TF, delimiter="," )




def generate_graph_embedding_src():
    flow = getTaxiFlow(normalization="none")
    row, column = flow.shape
    with open("multi-view-learning/taxi.od", 'w') as fout:
        for i in range(row):
            for j in range(column):
                if flow[i,j] > 0:
                    fout.write('{0} {1} {2}\n'.format(i, j, flow[i,j]))
                
            


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == 'generateTaxiFlow':
        print "Generate taxi flow"
        generateTaxiFlow()
    elif len(sys.argv) == 2 and sys.argv[1] == 'graphEmbedding':
        print "Generate graph embedding source"
        generate_graph_embedding_src()
    else:
        s = getTaxiFlow(leaveOut=-1, normalization="none")