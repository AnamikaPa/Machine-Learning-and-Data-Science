''' 
	IRIS
'''


import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

dataset = pd.read_csv('data/Iris', names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])
'''
print dataset.shape
print dataset.head(20)
print dataset.describe()
print dataset.groupby('class').size()


dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

dataset.hist()
plt.show()

scatter_matrix(dataset)
plt.show()
'''

array = dataset.values
X = array[:,0:4]
Y = array[:,4]

validation_size = 0.20
seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

print "---------------------------------------------------------------------------------------"

logreg = linear_model.LogisticRegression( C=1e5 )
logreg.fit(X_train,Y_train)

result = logreg.predict(X_validation)
print "Accuracy with Logistic Regression is ", accuracy_score(Y_validation, result)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)

result1 = knn.predict(X_validation)
print "Accuracy with K Neighbors Classifier is ",accuracy_score(Y_validation, result1)


dec = DecisionTreeClassifier()
dec.fit(X_train, Y_train)

result2 = dec.predict(X_validation)
print "Accuracy with Decision Tree Classifier is ",accuracy_score(Y_validation, result2)


lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)

result3 = lda.predict(X_validation)
print "Accuracy with Linear Discriminant Analysis is ",accuracy_score(Y_validation, result3)


gnb = GaussianNB()
gnb.fit(X_train, Y_train)

result4 = gnb.predict(X_validation)
print "Accuracy with GaussianNB is ",accuracy_score(Y_validation, result4)


svc = SVC()
svc.fit(X_train, Y_train)

result5 = svc.predict(X_validation)
print "Accuracy with Support Vector Machine is ",accuracy_score(Y_validation, result5)




