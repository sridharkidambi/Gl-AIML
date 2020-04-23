# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 18:10:12 2019

@author: Gaurav.Das
"""


import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve,roc_auc_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# os.chdir(r"D:\PERSONAL DATA\IMARTICUS\PYTHON Materials\ML")

def plotConfusionMatrix(cm, n,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    cm: confusion_matrix from sklearn.metrics
    n: number of dependent classes
    created by: gaurav.das2@gmail.com
    """
    if n == 2:
        classes = [0,1]
    elif n == 3:
        classes = [0,1,2]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, [0,1])
    plt.yticks(tick_marks, [0,1])

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def dtree_grid_search(X,y,nfolds): # nfolds = 5,4 -- 30*4 = 120 models
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(15, 30)} # pruning -- param_grid should be in the form of dictionary 
    # decision tree model
    dtree_model = DecisionTreeClassifier()
    #use gridsearch to val all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds) # 2 - modelName, paramters, cv
    #fit model to data
    dtree_gscv.fit(X, y) # X,y - train because you always fit on the train 
    #find score
    score = dtree_gscv.score(X, y) # score
    best_params = dtree_gscv.best_params_
    
    return best_params, score, dtree_gscv 

def RF_grid_search(X,y,nfolds):
    
    #create a dictionary of all values we want to test
    param = {'criterion':['gini','entropy'],'max_depth': np.arange(11, 19),
                  'n_estimators': [50, 100, 200, 300]}
    #randomForest model without gridSrearch
    rf = RandomForestClassifier() # without specifying any parameter
    #use gridsearch to val all values
    rf_gscv = GridSearchCV(estimator = rf, param_grid = param, cv=nfolds)
    #fit model to data
    rf_gscv.fit(X, y) # with grid search
    #find score
    score_gscv = rf_gscv.score(X, y) # with grid search
    
    return rf_gscv.best_params_, rf_gscv, score_gscv
    

