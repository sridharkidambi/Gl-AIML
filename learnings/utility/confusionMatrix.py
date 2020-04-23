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
    
