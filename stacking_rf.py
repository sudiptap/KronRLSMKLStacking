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
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model, datasets
from math import sqrt

#call matlab to generate predictions
print('Call successfully done')
avg = 0;
for i in range(5):
    dfpred = pd.read_table('C:/Users/Sudipta/Documents/DL/kronRLSMKStack/GPCR_stack/Folds/5fold/'+str(i+1)+'/predictions_all.txt',header=None, sep=',')
    dfpred_test = pd.read_table('C:/Users/Sudipta/Documents/DL/kronRLSMKStack/GPCR_stack/Folds/5fold/'+str(i+1)+'/predictions_test_all.txt',header=None, sep=',')
    dftrain = pd.read_csv('C:/Users/Sudipta/Documents/DL/kronRLSMKStack/GPCR_stack/Folds/5fold/'+str(i+1)+'/label_train_all.txt',header=None,sep=',')
    dftest = pd.read_csv('C:/Users/Sudipta/Documents/DL/kronRLSMKStack/GPCR_stack/Folds/5fold/'+str(i+1)+'/label_test_all.txt',header=None,sep=',')
    #print(dfpred_test.shape)
    #print(dftest)
    
    y1 = dftrain.as_matrix()
    y1_test = dftest.as_matrix()
    X = dfpred.as_matrix()
    X_test = dfpred_test.as_matrix()
    y = np.transpose(y1)
    y_test = np.transpose(y1_test)
    
    '''
    rng = np.random.RandomState(2)
    clf = IsolationForest(max_samples=4,contamination=0, random_state=rng)
    clf.fit(X)
    y_pred_test = clf.predict(X_test)
    n_error_test = y_pred_test[y_pred_test == -1].size
    print('test error : ',n_error_test)  
    print(y_pred_test)
    dfpred.close()
    dfpred_test.close()
    dftrain.close()'''
    
    '''
    clf = svm.OneClassSVM(kernel='rbf',max_iter=1000)
    clf.fit(X)
    y_pred_train = clf.predict(X)
    n_error_train = y_pred_train[y_pred_train == -1].size
    print('training error : ',n_error_train)  
    '''                           
                                
    #X = dfpred.as_matrix()
    #y1 = np.ones(72)#dftrain.as_matrix()
    #breg = LogisticRegression()
    #breg = svm.OneClassSVM(kernel='sigmoid',max_iter=1000)
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X, y)
    #breg = OneVsRestClassifier(GradientBoostingRegressor)
    y = np.transpose(y1)
    #print(X.shape)
    #print(y.shape)
    test_predict = clf.predict(X_test)
    print(test_predict)
    print(clf.score(X_test, y_test))
    #train_predict = breg.predict(X)
    #test_predict = breg.fit(X, y).decision_function(X_test)
    #for x in np.nditer(breg.coef_):
    #    print(x)
    
    #coeff_x = range(100)#np.arange(0, 100, 1)
    #coeff_y = breg.coef_
    #plt.bar(coeff_x,coeff_y)
    #plt.xticks(coeff_x)
    #plt.show()
    #print(sqrt(mean_squared_error(y_test, test_predict)))
    #for x in np.nditer(y_test):
    #    print(x)
    #for x in np.nditer(test_predict):
    #    print(x)
    #print(sqrt(mean_squared_error(y_test, test_predict))/(np.linalg.norm(y_test)))
    np.savetxt('test_out.txt',test_predict,delimiter=',')
    #print(confusion_matrix(y_test, test_predict))
    print('n_classes =' ,y.shape[1])
    
    n_classes = 2#y.shape[1]
    y_score = test_predict
    print('y_score.shape  ',y_score.shape)
    lw=2
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
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

    '''score = metrics.auc(test_predict, y_test, reorder=True)
    
    precision, recall, thresholds1 = precision_recall_curve(y_test, test_predict)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, test_predict, pos_label=1)
    print('fpr ',fpr)
    print('tpr ',tpr)
    print('auc ',metrics.auc(fpr, tpr))
    
    np.savetxt('test_out.txt',test_predict,delimiter=',')
    print(score)      
    print(precision)  
    print(recall)  
    print(thresholds1) '''
print('average over all : ', avg/5)    