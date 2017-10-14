import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

df = pd.read_csv('data/breast_cancer')
df.replace('?',-99999, inplace=True)
df.drop(['id'],1,inplace=True)
print df.head()

#features
X = np.array(df.drop(['class'],1))

#label
Y = np.array(df['class'])

#20% data for testing
x_train, x_test, y_train, y_test =  train_test_split(X, Y, test_size=0.2, random_state=0)

#clf = neighbors.KNeighborsClassifier()
#clf.fit(x_train,y_train)

#pickling means saving a model
#with open('pickle/k_nearest.pickle','wb') as f:
#	pickle.dump(clf,f)

pickle_in = open('pickle/k_nearest.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test, y_test)
print "Accuracy with K-Neighbors Classifier is:",accuracy

prediction = clf.predict(x_test)
print prediction
