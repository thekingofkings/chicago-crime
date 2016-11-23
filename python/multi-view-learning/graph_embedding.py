"""
Hongjian

Problem: use taxi flow as an independent view to predict crime.
Task:
    1. Get graph embedding representation of regions from taxi flow.
    2. Use leaveOneOut to test the error of this single view.
    3. Return a ML model on this single view, given training region ID.
"""

import numpy as np
import statsmodels.api as sm
from sklearn.cross_validation import LeaveOneOut

import sys
sys.path.append("../")

from FeatureUtils import retrieve_crime_count, generate_corina_features
from Crime import Tract



def get_graph_embedding_features(fn='taxi_all.txt'):
    """
    Get graph embedding vector, which is generated from LINE
    """
    ge = []
    with open(fn, 'r') as fin:
        fin.readline()
        for line in fin:
            ls = line.strip().split(" ")
            ge.append([float(i) for i in ls])
    ge = np.array(ge)
    ge = ge[np.argsort(ge[:,0])]
    return ge[:,1:]

    
    
def leaveOneOut_error(Y, X):
    """
    Use GLM model from python statsmodels library to fit data.
    Evaluate with leave-one-out setting, return the average of n errors.
    
    Input:    
    features    - a list features. ['all'] == ['demo', 'poi', 'geo', 'taxi']
    gwr_gamma   - the GWR weight matrx

    Output:
    error - the average error of k leave-one-out evaluation
    """
    errors = []
    errs_train = np.zeros(2)
    loo = LeaveOneOut(len(Y))
    X = sm.add_constant(X, prepend=False)
    for train_idx, test_idx in loo:    
        X_train, Y_train = X[train_idx], Y[train_idx]
        # Train NegativeBinomial Model from statsmodels library
        glm = sm.GLM(Y_train, X_train, family=sm.families.NegativeBinomial())
        nbm = glm.fit()
        ybar = nbm.predict(X[train_idx])
        er_train = np.mean(np.abs(ybar - Y[train_idx]))
        errs_train += er_train, er_train / np.mean(Y[train_idx])
#        print er_train, er_train / np.mean(Y[train_idx])
        ybar = nbm.predict(X[test_idx])
        errors.append(np.abs(ybar - Y[test_idx]))
    print errs_train / len(Y)
    return np.mean(errors), np.mean(errors / Y), np.mean(errors) / np.mean(Y)
    
    
    
    
    
def predict_crime_with_embedding():
    ge = get_graph_embedding_features("taxi_all.txt")
    
    y_cnt = retrieve_crime_count(2010)
    demo = generate_corina_features()
    population = demo[1][:,0].reshape(demo[1].shape[0], 1)
    y = y_cnt / population * 10000
    
    er = leaveOneOut_error(y, ge)
    print er
    return er
    
    


def CA_clustering_with_embedding():
    ge = get_graph_embedding_features("geo_all.txt")
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=6, max_iter=100).fit(ge)
    for idx, lab in enumerate(kmeans.labels_):
        print idx+1, lab
    
    colorMaps = ['blue', 'red', 'g', 'c', 'y', 'm', 'k', 'w']
    cas = Tract.createAllCAObjects()
    import matplotlib.pyplot as plt
    import descartes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k in cas:
        cak = cas[k].polygon
        ax.add_patch(descartes.PolygonPatch(cak, fc=colorMaps[kmeans.labels_[k-1]]))
        ax.annotate(str(k), [cak.centroid.x, cak.centroid.y])
    ax.axis('equal')
    fig.show()
    
    return kmeans, cas

    
    

if __name__ == '__main__':
    
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == 'leaveOneOut':
        predict_crime_with_embedding()
    else:
        kmeans, cas = CA_clustering_with_embedding()
    
