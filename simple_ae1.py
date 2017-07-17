import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.models import Sequential
from keras import regularizers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

train_file_name = 'example_train_3.txt'
test_file_name = 'example_test_3.txt'
encoding_dim = 60
temp_train = pd.read_table(train_file_name)
temp_test = pd.read_table(test_file_name)
output_dim = temp_train.shape[1]-1
print(temp_train)

#input_img = pd.read_table(dataset_name)
train_img = np.loadtxt(train_file_name,'\t',skiprows=1, usecols=range(1,output_dim))
test_img = np.loadtxt(test_file_name,'\t',skiprows=1, usecols=range(1,output_dim))
print(train_img.shape)
input_train_tf = train_img
input_test_tf = test_img
#input = input_img.drop(input_img.columns[0], axis=1).values
#input = input_img.values
#input_tf = tf.Variable(input, dtype='float32')
model = Sequential()
model.add(Dense(encoding_dim, activation='relu', name='encoder', input_dim=output_dim-1))
model.add(Dense(output_dim-1, activation='relu', name='decoder'))
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
#train, test = train_test_split(input_tf, train_size=0.8)
#kfold = KFold(n_splits=5, shuffle=True, random_state=4242)
cvscores = []
aucs = []
#X = 
'''for train,test in kfold.split(input_tf):
    model.fit(input_tf[train], input_tf[train], epochs=100, batch_size=10, verbose=0)
    scores = model.evaluate(input_tf[test], input_tf[test], verbose=0)
    decoded_img = model.predict(input_tf[test])
    decoded_flattened = decoded_img.flatten()
    truth_flattened = input_tf[test].flatten()
    precision, recall, thresholds = precision_recall_curve(truth_flattened, decoded_flattened)
    area = metrics.auc(recall, precision)
    print('AUPR ==> ',area)
    #print(decoded_flattened.shape)
    #print(truth_flattened.shape)
    #print(decoded_img)
    #print(model.metrics_names[1], scores[1]*100)
    cvscores.append(scores[1]*100)
    aucs.append(area * 100)
print(np.mean(cvscores),np.std(cvscores),np.mean(aucs))'''

model.fit(input_train_tf, input_train_tf, epochs=50, batch_size=10, verbose=0)
scores = model.evaluate(input_test_tf, input_test_tf, verbose=0)
decoded_img = model.predict(input_test_tf)
decoded_flattened = decoded_img.flatten()
truth_flattened = input_test_tf.flatten()
precision, recall, thresholds = precision_recall_curve(truth_flattened, decoded_flattened)
area = metrics.auc(recall, precision)
print('AUPR ==> ',area)
#print(decoded_flattened.shape)
#print(truth_flattened.shape)
#print(decoded_img)
#print(model.metrics_names[1], scores[1]*100)
cvscores.append(scores[1]*100)
aucs.append(area * 100)
print(np.mean(cvscores),np.std(cvscores),np.mean(aucs))


  

