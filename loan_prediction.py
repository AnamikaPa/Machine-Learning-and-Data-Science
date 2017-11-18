import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale


X_train = pd.read_csv('data/loan_prediction-1/X_train.csv')
Y_train = pd.read_csv('data/loan_prediction-1/Y_train.csv')

X_test = pd.read_csv('data/loan_prediction-1/X_test.csv')
Y_test = pd.read_csv('data/loan_prediction-1/Y_test.csv')

print X_train.head()

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History']],Y_train)
print " "
print "Accuracy: ",accuracy_score(Y_test,knn.predict(X_test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]))

min_max=MinMaxScaler()
X_train_minmax=min_max.fit_transform(X_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_minmax=min_max.fit_transform(X_test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_minmax,Y_train)
print " "
print "Accuracy with normalization: ",accuracy_score(Y_test,knn.predict(X_test_minmax))

X_train_scale=scale(X_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_scale=scale(X_test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scale,Y_train)
print " "
print "Accuracy with Scaleling: ",accuracy_score(Y_test,knn.predict(X_test_scale))

