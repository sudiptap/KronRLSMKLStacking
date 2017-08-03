import os
import sys

#from sklearn import model_selection
#from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, preprocessing
#from stacked_generalization.lib.stacking import StackedRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
#from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from math import sqrt
from sklearn.svm import NuSVC
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
#from sklearn.model_selection import GridSearchCV, cross_val_score

#call matlab to generate predictions
print('Call successfully done')

foldername = 'C:/Users/Sudipta/Documents/DL/kronRLSMKStack/New_Results_50FoldValidation/E/'
avg = 0;
for i in range(5):
    dfpred = pd.read_table(foldername+str(i+1)+'/predictions_all.txt',header=None, sep=',')
    dfpred_test = pd.read_table(foldername+str(i+1)+'/predictions_test_all.txt',header=None, sep=',')
    dftrain = pd.read_csv(foldername+str(i+1)+'/label_train_all.txt',header=None,sep=',')
    dftest = pd.read_csv(foldername+str(i+1)+'/label_test_all.txt',header=None,sep=',')
    
    y1 = dftrain.as_matrix()
    y1_test = dftest.as_matrix()    
    with open('tmpfile','r') as f:
        a = f.read()
    ind_list = list(map(int,a.split()))
    TOP = 100
    top_ind = ind_list[-TOP:]    

    X = dfpred.iloc[:,top_ind].as_matrix()
    X_test = dfpred_test.iloc[:,top_ind].as_matrix()

    y = np.transpose(y1)
    y_test = np.transpose(y1_test)
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    scores = ['precision', 'recall']
    
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        
        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring='%s_macro' % score)
        
        #print(X.shape)
        print(y.ravel().shape)
        y = y.ravel()
        clf.fit(X, y)
        
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        y_score = y_pred
        y_test = y_true
    
        ''' 
        breg = OneVsRestClassifier(svm.SVC(kernel='rbf', cache_size=200, coef0=1.0, verbose=True, degree=10, probability=True,random_state=np.random.RandomState(0), max_iter=10000, shrinking=True, C=1, tol=0.00001, decision_function_shape='ovr'))
         
        y = np.transpose(y1)   
        
        breg.fit(X, y)
        
        test_predict = breg.fit(X, y).decision_function(X_test)
        test_predict = test_predict.reshape(test_predict[0],1)
        
        np.savetxt('test_out.txt',test_predict,delimiter=',') '''   
        
        n_classes = 1#y.shape[1]
        #y_score = test_predict
        print('y_score.shape  ',y_score.shape)
        lw=2
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
        
        # Compute Precision-Recall and plot curve
        precision = dict()
        recall = dict()
        average_precision = dict()
        
        print(y_test.shape)
        #print(np.reshape(y_score, (y.score.shape(0),1)))
        y_score = np.reshape(y_score, (y_score.shape[0],1))
        print(y_score.shape)
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
        
        # Compute micro-average ROC curve and ROC area
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
            y_score.ravel())
        average_precision["micro"] = average_precision_score(y_test, y_score,
                                                             average="micro")
        
        avg = avg + average_precision[0]
        print('sum avg is :', avg)
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
        
        # Plot Precision-Recall curve for each class
        plt.clf()
        plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
                 label='micro-average Precision-recall curve (area = {0:0.2f})'
                       ''.format(average_precision["micro"]))
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=lw,
                     label='Precision-recall curve of class {0} (area = {1:0.2f})'
                           ''.format(i, average_precision[i]))
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(loc="lower right")
        plt.show()

    
print('average over all : ', avg/5)    
