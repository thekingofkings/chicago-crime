#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Preliminary study on the relation between taxi flow and crime rate difference.


Created on Sat May 20 16:30:33 2017

@author: hxw186
"""


from FeatureUtils import retrieve_crime_count, generate_corina_features
from taxiFlow import getTaxiFlow
import matplotlib.pyplot as plt
import pickle
from multi_view_learning.multi_view_prediction import similarityMatrix



def generate_point(F, Y):
    x = []
    y = []
    
    xp = []
    yp = []
    lp = []
    
    for i in range(77):
        for j in range(77):
            if i == j:
                continue
            if F[i][j] < 5000 and F[i][j] > 1:
                x.append(F[i][j])
                y.append(Y[i] - Y[j])
            else:
                xp.append(F[i][j])
                yp.append(Y[i] - Y[j])
                lp.append("{0}-{1}".format(i+1,j+1))
    return x, y, xp, yp, lp


demo = generate_corina_features()
y_cnt = retrieve_crime_count(2013)

population = demo[1][:,0].reshape(demo[1].shape[0], 1)
Y = y_cnt / population * 10000


F = getTaxiFlow(normalization="none")

x, y, xp, yp, lp = generate_point(F, Y)

f = plt.figure()
plt.scatter(x, y, color='red')
#a = plt.scatter(xp, yp, color='blue')
#for i in range(len(lp)):
#    a.axes.annotate(lp[i], xy=(xp[i], yp[i]))
plt.show()



with open("multi_view_learning/CAflowFeatures.pickle") as fin:
    mf = pickle.load(fin)
    line = pickle.load(fin)
    dwt = pickle.load(fin)
    dws = pickle.load(fin)
    hdge = pickle.load(fin)
    


for h in range(24):
    Fn = similarityMatrix(hdge[h])
    x, y, xp, yp, lp = generate_point(Fn, Y)
    f = plt.figure()
    plt.scatter(x, y, color='red')
    plt.show()