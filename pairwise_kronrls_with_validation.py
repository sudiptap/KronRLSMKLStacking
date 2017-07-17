import os
import sys

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, preprocessing
#from stacked_generalization.lib.stacking import StackedRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import KFold


dfpred = pd.read_table('C:/Users/Sudipta/Documents/DL/kronRLSMKStack/predictions_all.txt',header=None, sep=',')
dfpred_test = pd.read_table('C:/Users/Sudipta/Documents/DL/kronRLSMKStack/predictions_test_all.txt',header=None, sep=',')
dftrain = pd.read_csv('C:/Users/Sudipta/Documents/DL/kronRLSMKStack/label_train_all.txt',header=None,sep=',')
dftest = pd.read_csv('C:/Users/Sudipta/Documents/DL/kronRLSMKStack/label_test_all.txt',header=None,sep=',')
#print(dfpred_test.shape)
#print(dftest)

y1 = dftrain.as_matrix()
y1_test = dftest.as_matrix()
X = dfpred.as_matrix()
X_test = dfpred_test.as_matrix()
y = np.transpose(y1)
y_test = np.transpose(y1_test)


#KFold trial
kf = KFold(n_splits=2)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_validation = X[train_index], X[test_index]
    y_train, y_validation = y[train_index], y[test_index]
                       

    all_aupr = [];
    bsf = 0
    avg = 0
    bsf_ind = -1
    n_classes = 1
    lw=2
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    for kernel_id in range(X_train.shape[1]):
        y_score = X_validation[:,kernel_id].reshape(X_validation.shape[0], 1)
        print('y_score.shape  ',y_score.shape)
        # Compute Precision-Recall and plot curve
        precision = dict()
        recall = dict()
        average_precision = dict()
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_validation[:, i],
                                                                y_score[:, i])
            average_precision[i] = average_precision_score(y_validation[:, i], y_score[:, i])
    
        if average_precision[0] >= bsf:
            bsf = average_precision[0]
            bsf_ind = kernel_id
        avg = avg + average_precision[0]
        all_aupr.append(average_precision[0])
        # Plot Precision-Recall curve
        plt.clf()
        plt.plot(recall[0], precision[0], lw=lw, color='navy',
                 label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
        plt.legend(loc="lower left")
        plt.show()
    
    print('best is ',bsf,' and corresponding id is ', bsf_ind)
    print('avg is ',avg/X.shape[1])   
        
    for p in all_aupr:
        print(p)
  