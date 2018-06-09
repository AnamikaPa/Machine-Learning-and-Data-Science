"""
Download data from https://www.kaggle.com/mlg-ulb/creditcardfraud/data
"""

import numpy as np 
import pandas as pd 
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

text = pd.read_csv('creditcard.csv')#,nrows = 10)
text = text.sample(frac=1).reset_index(drop=True)

labels = text['Class']
features = text.drop(['Class'],axis=1)

x_train, x_test, y_train, y_test =  train_test_split(features, labels, test_size=0.30, random_state=0)

clf = DecisionTreeClassifier(random_state=0, min_samples_split=2, min_samples_leaf=1)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print accuracy
