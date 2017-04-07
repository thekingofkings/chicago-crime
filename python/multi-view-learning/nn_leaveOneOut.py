#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Tensor flow NN model for leave-one-out evaluation.

Created on Fri Apr  7 14:25:01 2017

@author: hxw186
"""


import tensorflow as tf
import numpy as np


import sys
sys.path.append("../")

from feature_evaluation import build_features


def leaveOneOut_error(Y, D, P, Tf, Yt, Gd, Yg, features=['all'], gwr_gamma=None, taxi_norm="bydestination"):
    """
    Use GLM model from python statsmodels library to fit data.
    Evaluate with leave-one-out setting, return the average of n errors.
    
    Input:    
    features    - a list features. ['all'] == ['demo', 'poi', 'geo', 'taxi']
    gwr_gamma   - the GWR weight matrx. TODO

    Output:
    error - the average error of k leave-one-out evaluation
    """
   
    errors = []
    for k in range(len(Y)):
        with tf.Graph().as_default():
            X_train, X_test, Y_train, Y_test = build_features(Y, D, P, Tf, Yt, Gd, Yg, k, features, taxi_norm)
             # build the TF nn model
            F1 = X_train.shape[1]
            x1 = tf.placeholder(tf.float32, [None, F1], name="numeric_features_set1")
            y = tf.placeholder(tf.float32, [None, 1], name="label")
            
            W = tf.Variable(tf.random_normal([F1]), name="weight")
            b = tf.Variable(tf.random_normal([1]), name="bias")
            
            y_est = tf.add(tf.reduce_sum(tf.multiply(x1, W)), b)
            
#            h1 = tf.layers.dense(inputs=x1, units=F1/2, activation=tf.nn.relu, use_bias=True,
#                                name="reduce_half", reuse=None)
#            y_est = tf.layers.dense(inputs=x1, units=1, activation=None, use_bias=True,
#                                name="reg_pred", reuse=None)
            
            objective = tf.reduce_mean(tf.squared_difference(y, y_est))
            
            train_step = tf.train.GradientDescentOptimizer(0.1).minimize(objective)
            tf_mae = tf.reduce_mean(tf.abs(y - y_est))
            
            
            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
    
            train_step.run(feed_dict={x1: X_train, y: Y_train[:,None]})
            yarray = np.array(Y_test).reshape((1,1))
            mae = tf_mae.eval(feed_dict={x1: X_test[None,:], y: yarray})
            errors.append(mae)
        
    
    return np.mean(errors), np.mean(errors) / np.mean(Y)
