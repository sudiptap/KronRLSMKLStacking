# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
# fix random seed for reproducibility
np.random.seed(7)
# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
y = dataset[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)




# create model
model = Sequential()
model.add(Dense(10, input_dim=100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, verbose=1, validation_split=0.1, epochs=500, batch_size=10)
model.fit(X_train, y_train, verbose=1, epochs=300, batch_size=10)
# evaluate the model
#scores = model.evaluate(X, y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


predictions = model.predict(X_test)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)



real = y_test.reshape((77,1))
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(real[:, i],
                                                        predictions[:, i])
    average_precision[i] = average_precision_score(real[:, i], predictions[:, i])
	
precision["micro"], recall["micro"], _ = precision_recall_curve(real.ravel(),
    predictions.ravel())
average_precision["micro"] = average_precision_score(real, predictions,
                                                     average="micro")
													 
													 
													 
													 
													 
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(real[i, :],
                                                        predictions[i, :])
    average_precision[i] = average_precision_score(real[i, :], predictions[i, :])
	
precision["micro"], recall["micro"], _ = precision_recall_curve(real.ravel(),
    predictions.ravel())
average_precision["micro"] = average_precision_score(real, predictions,
                                                     average="micro")
													 
													 
													 
# Plot Precision-Recall curve
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2


plt.clf()

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




dfpred = pd.read_table('1'+'/predictions_all.txt',header=None, sep=',')
dfpred_test = pd.read_table('1'+'/predictions_test_all.txt',header=None, sep=',')
dftrain = pd.read_csv('1'+'/label_train_all.txt',header=None,sep=',')
dftest = pd.read_csv('1'+'/label_test_all.txt',header=None,sep=',')


y1 = dftrain.as_matrix()
y1_test = dftest.as_matrix()
X = dfpred.as_matrix()
X_test = dfpred_test.as_matrix()
y = np.transpose(y1)
y_test = np.transpose(y1_test)