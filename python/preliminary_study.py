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
import matplotlib
matplotlib.rc('pdf', fonttype=42)


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
            if F[i][j] < 4000 and F[i][j] >= 0:
                if i == 31 or j == 31:
                    xp.append(F[i][j])
                    yp.append(Y[i] - Y[j])
                    lp.append("{0}-{1}".format(i+1,j+1))
                else:
                    x.append(F[i][j])
                    y.append(Y[i] - Y[j])
    return x, y, xp, yp, lp




def plot_embedding_cases(Y):
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




demo = generate_corina_features()
y_cnt = retrieve_crime_count(2013)

population = demo[1][:,0].reshape(demo[1].shape[0], 1)
Y = y_cnt / population * 10000


F = getTaxiFlow(normalization="none")

x, y, xp, yp, lp = generate_point(F, Y)

plt.rc("axes", linewidth=2)
f = plt.figure(figsize=(8,6))
plt.scatter(x, y, s=16)

plt.plot([-100, -100, 3500, -100], [3000, -3000, 0, 3000], linewidth=2, color='blue')
plt.scatter(xp, yp, color='red', s=28)
plt.xlabel("Taxi flow from $r_i$ to $r_j$", fontsize=20)
plt.ylabel("Crime rate difference $y_i - y_j$", fontsize=20)
#for i in range(len(lp)):
#    a.axes.annotate(lp[i], xy=(xp[i], yp[i]))

plt.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig("crime-flow-preliminary.pdf")




