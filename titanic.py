import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np 
from sklearn.cluster import KMeans,MeanShift
from sklearn import preprocessing
import pandas as pd 
import pickle
 
df = pd.read_excel('data/titanic.xls')
original_df = pd.DataFrame.copy(df)

df.drop(['body','name','ticket','boat'], 1, inplace=True)
print df.head()

df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)


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
print df.head()

x = np.array(df.drop(['survived'],1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])

# K-Means classifier

#clf = KMeans(n_clusters=2)
#clf.fit(x)

#with open('pickle/k_means_titanic.pickle','wb') as f:
#	pickle.dump(clf,f)

pickle_in = open('pickle/k_means_titanic.pickle','rb')
clf = pickle.load(pickle_in)

correct = 0
for i in range(len(x)):
	predict_me = np.array(x[i].astype(float))
	predict_me = predict_me.reshape(-1, len(predict_me))
	prediction = clf.predict(predict_me)
	if prediction[0] == y[i]:
		correct +=1

print "Accuracy with K Means:", float(correct)/len(x)

# Hierarchical clustering

#clf = MeanShift()
#clf.fit(x)

#with open('pickle/mean_shift_titanic.pickle','wb') as f:
#	pickle.dump(clf,f)

pickle_in = open('pickle/mean_shift_titanic.pickle','rb')
clf = pickle.load(pickle_in)


labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(x)):
	original_df['cluster_group'].iloc[i] = labels[i]

n_clusters = len(np.unique(labels))

#finding each cluster's survival rate
survival_rates = {}
for i in range(n_clusters):
	temp_df = original_df[(original_df['cluster_group']==float(i))]
	survival_cluster = temp_df[(temp_df['survived']==1)]
	survival_rate = float(len(survival_cluster))/len(temp_df)
	survival_rates[i] = survival_rate

for i in range(0,len(survival_rates)):
	print "Survival rate of group ",i,"is:",survival_rates[i]
