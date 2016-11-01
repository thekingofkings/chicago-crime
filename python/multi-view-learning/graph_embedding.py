 
import numpy as np
import statsmodels.api as sm
from sklearn.cross_validation import LeaveOneOut

import sys
sys.path.append("../")

from FeatureUtils import retrieve_crime_count, generate_corina_features
from Crime import Tract



def get_graph_embedding_features(hasConstant=True):
    """
    Get graph embedding vector, which is generated from LINE
    """
    ge = []
    with open('vec_all.txt', 'r') as fin:
        fin.readline()
        for line in fin:
            ls = line.strip().split(" ")
            ge.append([float(i) for i in ls])
    ge = np.array(ge)
    if hasConstant:
        ge[:,0] = 1
        return ge
    else:
        return ge[:, 1:]
    
    


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
    for train_idx, test_idx in loo:
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        # Train NegativeBinomial Model from statsmodels library
        nbm = sm.GLM(Y_train, X_train, family=sm.families.NegativeBinomial())
        nb_res = nbm.fit()
        ybar = nbm.predict(nb_res.params, X_train)
        er_train = np.mean(np.abs(ybar - Y_test))
        errs_train += er_train, er_train / np.mean(Y_test)
#        print er_train, er_train / np.mean(Y_test)
        
        ybar = nbm.predict(nb_res.params, X_test)
        errors.append(np.abs(ybar - Y_test))
    print errs_train / len(Y)
    return np.mean(errors), np.mean(errors) / np.mean(Y)
    
    
    
    
    
def predict_crime_with_embedding():
    ge = get_graph_embedding_features()
    
    y_cnt = retrieve_crime_count(2010)
    demo = generate_corina_features()
    population = demo[1][:,0].reshape(demo[1].shape[0], 1)
    y = y_cnt / population * 10000
    
    er = leaveOneOut_error(y, ge)
    print er
    return er
    
    


def CA_clustering_with_embedding():
    ge = get_graph_embedding_features(hasConstant=False)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, max_iter=100).fit(ge)
    for idx, lab in enumerate(kmeans.labels_):
        print idx+1, lab
    
    colorMaps = ['blue', 'red', 'g', 'c', 'w']
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
    