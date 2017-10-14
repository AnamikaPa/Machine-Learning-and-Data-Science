import pandas as pd
import quandl
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import math, datetime, time
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

quandl.ApiConfig.api_key = 'UVA-xzvLy3qBreZsh3Qf'
df = quandl.get_table('WIKI/PRICES')

#df = quandl.get('WIKI/GOOGL')



df['date'] = pd.to_datetime(df.date)
df = df.sort_values(by='date')

df = df[(df['date'].dt.year > 1998)]

'''
scatter_matrix(df)
plt.show()
'''

#selecting only meaningfull column
df = df[['date','adj_open','adj_high','adj_low','adj_close','adj_volume']]

#making new column of percentage change
df['HL_PCT'] = (df['adj_high'] - df['adj_close'])/df['adj_close'] *100

#daily percent change
df['PCT_change'] = (df['adj_close'] - df['adj_open'])/df['adj_open'] *100

df = df[['date','adj_close','HL_PCT','PCT_change','adj_volume']]

#label
forecast_col = 'adj_close'

#fill not applicable data with -99999, coz in ml we cannot able to work with NA data
df.fillna(-99999 , inplace=True)

#data for prediction
forecast_out = int(math.ceil(0.001*len(df)))

#what we r going here is, we have present data and we make its lable as (0.01*len(df))days in future the value Adj. Close
df['label'] = df[forecast_col].shift(-forecast_out)


df.dropna(inplace=True)

#features
x = np.array(df[['adj_close','HL_PCT','PCT_change','adj_volume']]) 

#Standardize a dataset along any axis
x = preprocessing.scale(x) 
x_lately = x[-forecast_out:] 
x = x[:-forecast_out]

#label
y = np.array(df['label'])
y_lately = y[-forecast_out:] 
y = y[:-forecast_out]


#20% data for testing
x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.2, random_state=0)


#defining Classifier

#Linear Regression
# n_jobs means no. of threads running prallel, if its value is -1 means it will run as many as threads possible


#clf = LinearRegression(n_jobs=10)
#clf.fit(x_train, y_train)

#pickling means saving a model
#with open('linearregression.pickle','wb') as f:
#	pickle.dump(clf,f)

pickle_in = open('pickle/linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test,y_test)
forecast_set = clf.predict(x_lately)
print "Accuracy with Linear Regression : %f"% accuracy

'''
#Support Vector Machine
clf2 = svm.SVR(kernel='poly')
clf2.fit(x_train, y_train)
accuracy1 = clf2.score(x_test,y_test)
print "Accuracy with Support Vector Machine : %f"% accuracy1
'''

#initialising
df['Forecast'] = np.nan

#this is we doing because we didn't know the predicted value belong to which date
last_date = df['date'][df.index[-1]]
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix).date()
	next_unix += one_day
	df.loc[next_date] = [next_date, np.nan, np.nan, np.nan, np.nan, np.nan]+[i]

print df.tail()

#df['adj_close'].plot()
plt.plot(df.date,df.Forecast)
plt.plot(df.date,df.adj_close)
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

