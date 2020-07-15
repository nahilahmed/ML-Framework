# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:02:16 2020

@author: nahil
"""


from sklearn import ensemble
from sklearn import tree

models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    "rf": ensemble.RandomForestClassifier(),
}

