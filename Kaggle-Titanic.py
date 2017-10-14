import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np 
from sklearn.cluster import KMeans,MeanShift
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd 
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


df = pd.read_csv('data/train.csv')
df.drop(['Name','Cabin','PassengerId'],1,inplace=True)

print df.head()

df_test = pd.read_csv('data/test.csv')
idd = df_test['PassengerId']
df_test.drop(['Name','Cabin','PassengerId'],1,inplace=True)

df_result = pd.read_csv('data/gender_submission.csv')
df_result.drop('PassengerId',1,inplace=True)


median_age = df['Age'].median()
median_age1 = df_test['Age'].median()

df['Age'].fillna(median_age,inplace=True)
df_test['Age'].fillna(median_age1,inplace=True)

#sex = {'male':1, 'female':0}
#df['Sex'] = df['Sex'].map(sex)

df['Embarked'].fillna('S',inplace=True)
df_test['Embarked'].fillna('S',inplace=True)


def handle_non_numerical_data(df):
	columns = df.columns.values

	for column in columns:
		text_digit_vals = {}
		def convert_to_int(val):
			return text_digit_vals[val]

		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x=0

			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x+=1

			df[column] = list(map(convert_to_int, df[column]))

	return df

df = handle_non_numerical_data(df)
df_test = handle_non_numerical_data(df_test)

x = np.array(df.drop(['Survived'],1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['Survived'])

a = np.mean(df_test[~np.isnan(df_test)])
df_test.fillna(a,inplace=True)


x_test = np.array(df_test.astype(float))
x_test = preprocessing.scale(x_test)

y_result = np.array(df_result.astype(float))


#---------------------------------------K-Means-----------------------------------------
clf = KMeans(n_clusters=2)
clf.fit(x)

#with open('k_means_titanic.pickle','wb') as f:
#	pickle.dump(clf,f)

#pickle_in = open('k_means_titanic.pickle','rb')
#clf = pickle.load(pickle_in)

a = clf.predict(x_test)

correct = 0
for i in range(len(x_test)):
	if a[i] == y_result[i]:
		correct +=1

re = float(correct)/len(x_test)

if(re<1-re):
	re = 1-re

print "Accuracy with K Means:", re

a = pd.DataFrame(a)
a.set_index(idd,inplace=True)
a.columns = ['Survived']
prediction1 = pd.DataFrame(a).to_csv('data/prediction1.csv')

#---------------------------------------SVM-----------------------------------------

svm_clf = svm.SVC()
svm_clf.fit(x,y)
y_prediction = svm_clf.predict(x_test)

print "Accuarcy with SVM: ",accuracy_score(y_result,y_prediction)

y_prediction_df = pd.DataFrame(y_prediction)
y_prediction_df.set_index(idd,inplace=True)
y_prediction_df.columns = ['Survived']

prediction = pd.DataFrame(y_prediction_df).to_csv('data/prediction.csv')

#---------------------------------------Random Forest-----------------------------------------

forest = RandomForestClassifier(n_estimators = 1000)
forest.fit(x,y)

forest_pred = forest.predict(x_test)
print "Accuarcy with RandomForest: ",accuracy_score(y_result,forest_pred)

forest_pred_df = pd.DataFrame(forest_pred)
forest_pred_df.set_index(idd,inplace=True)
forest_pred_df.columns = ['Survived']

prediction = pd.DataFrame(y_prediction_df).to_csv('data/prediction2.csv')
