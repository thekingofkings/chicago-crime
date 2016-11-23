#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:16:43 2016

@author: hxw186

Evaluate multi-view learning framework.
"""

import numpy as np
import statsmodels.api as sm
from sklearn.cross_validation import LeaveOneOut

import sys
sys.path.append("../")

from graph_embedding import get_graph_embedding_features
from feature_evaluation import extract_raw_samples



def generate_raw_samples():
    Y, D, P, Tf, Gd = extract_raw_samples()
    T = get_graph_embedding_features()


def NBmodel(train_idx, Y, X):
    nbm = sm.GLM(Y[train_idx], X[train_idx], family=sm.families.NegativeBinomial())
    nb_res = nbm.fit()
    return nb_res
    
   
    
def mvl_fuse_function(models, Y_test):
    