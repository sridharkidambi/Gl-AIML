import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from sklearn.feature_extraction.text import CountVectorizer  #DT does not take strings as input for the model fit step....
from IPython.display import Image  
#import pydotplus as pydot
from sklearn import tree
from os import system
from dataclasses import replace
import os
import sys;
# Ensemble Learning - Bagging
from sklearn.ensemble import BaggingClassifier

# Ensemble Learning - AdaBoosting
from sklearn.ensemble import AdaBoostClassifier

# Ensemble Learning - GradientBoost
from sklearn.ensemble import GradientBoostingClassifier

# Ensemble RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from confusionMatrix import plotConfusionMatrix,dtree_grid_search,RF_grid_search;
# from confusionMatrix import dtree_grid_search;
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import confusion_matrix, classification_report, roc_curve,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost  import XGBClassifier 