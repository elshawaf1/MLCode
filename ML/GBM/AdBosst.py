'''
Write a Python function adaboost_fit that implements the fit method for an AdaBoost classifier. The function should take in a 2D numpy 
array X of shape (n_samples, n_features) representing the dataset, a 1D numpy array y of shape (n_samples,) representing the labels, and 
an integer n_clf representing the number of classifiers. 

The function should :

1-initialize sample weights,
2-find the best thresholds for each feature,
3-calculate the error,
4-update weights, 
5-return a list of classifiers with their parameters.

Example:
Input:
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 1, -1, -1])
    n_clf = 3

    clfs = adaboost_fit(X, y, n_clf)
    print(clfs)

Output:
(example format, actual values may vary):
    # [{'polarity': 1, 'threshold': 2, 'feature_index': 0, 'alpha': 0.5},
    #  {'polarity': -1, 'threshold': 3, 'feature_index': 1, 'alpha': 0.3},
    #  {'polarity': 1, 'threshold': 4, 'feature_index': 0, 'alpha': 0.2}]
Reasoning:
The function fits an AdaBoost classifier on the dataset X with the given labels y and number of classifiers n_clf. It returns a list of classifiers with their parameters, including the polarity, threshold, feature index, and alpha values
'''

import numpy as np
import math


def adaboost_fit(X, y, n_clf):
	n_samples, n_features = np.shape(X)
	w = np.full(n_samples, (1 / n_samples))
	clfs = []
	
	return clfs


X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([1, 1, -1, -1])
n_clf = 3

clfs = adaboost_fit(X, y, n_clf)
print(clfs)