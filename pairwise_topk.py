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

dfpred = pd.read_table('predictions_all.txt',header=None, sep=',')
dfpred_test = pd.read_table('predictions_test_all.txt',header=None, sep=',')
dftrain = pd.read_csv('label_train_all.txt',header=None,sep=',')
dftest = pd.read_csv('label_test_all.txt',header=None,sep=',')
#print(dfpred_test.shape)
#print(dftest)

y1 = dftrain.as_matrix()
y1_test = dftest.as_matrix()
X = dfpred.as_matrix()
X_test = dfpred_test.as_matrix()
y = np.transpose(y1)
y_test = np.transpose(y1_test)

                       
'''                            
#X = dfpred.as_matrix()
#y1 = np.ones(72)#dftrain.as_matrix()
#breg = svm.OneClassSVM(kernel='rbf',max_iter=1000)
breg = OneVsRestClassifier(svm.SVC(kernel='linear',  probability=True,random_state=np.random.RandomState(0), max_iter=1000))
#breg = OneVsRestClassifier(GradientBoostingRegressor)
y = np.transpose(y1)
#print(X.shape)
#print(y.shape)
breg.fit(X, y)
#test_predict = breg.predict(X_test)
test_predict = breg.fit(X, y).decision_function(X_test)
#print(sqrt(mean_squared_error(y_test, test_predict)))
for x in np.nditer(y_test):
    print(x)
for x in np.nditer(test_predict):
    print(x)
print(sqrt(mean_squared_error(y_test, test_predict))/(np.linalg.norm(y_test)))
np.savetxt('test_out.txt',test_predict,delimiter=',')
#print(confusion_matrix(y_test, test_predict))'''

K =10 

bsf = 0
bsf_topk = 0.0
avg = 0
avg_topk = 0.0
bsf_ind = -1
n_classes = 1
lw=2
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
for kernel_id in range(X.shape[1]):
    y_score = X_test[:,kernel_id].reshape(X_test.shape[0], 1)
    print('y_score.shape  ',y_score.shape)
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
        
        # sort score and find how many 1's are captured in top k scores
        pair = [y_score[:,i], y_test[:,i]]
        pair = pd.DataFrame(pair)
        pair = pair.sort(0, axis=1)
        pair = pair.as_matrix()
        print(pair)
        sums = (pair[1,:] == 1).sum()
        print(sums)
        cnt = 0.0
        for e in pair[1, -K:]:
            if e == 1.0:
                cnt = cnt + 1 
        print(cnt)
        topk_ratio = cnt / sums
        print(topk_ratio)
        avg_topk = avg_topk + topk_ratio
        if bsf_topk < topk_ratio:
            bsf_topk = topk_ratio

    if average_precision[0] >= bsf:
        bsf = average_precision[0]
        bsf_ind = kernel_id
    avg = avg + average_precision[0]

    # Plot Precision-Recall curve
#    plt.clf()
#    plt.plot(recall[0], precision[0], lw=lw, color='navy',
#             label='Precision-Recall curve')
#    plt.xlabel('Recall')
#    plt.ylabel('Precision')
#    plt.ylim([0.0, 1.05])
#    plt.xlim([0.0, 1.0])
#    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
#    plt.legend(loc="lower left")
#    plt.show()

print('best is ',bsf,' and corresponding id is ', bsf_ind)
print('avg is ',avg/X.shape[1])   
print('k is ',K,'and')
print('best top k is ', bsf_topk)
print('avg top k is ', avg_topk/100)
    
    
