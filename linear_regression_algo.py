# first find the BEST FIT line ( y=mx+b )
# m = ( (mean of x)*(mean of y) - (mean of xy) )/( (mean of x)^2 - (mean of x^2) )

# b = (mean of y) - m*(mean of x)

from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

df = pd.read_excel('data/regression.xls') 
print df

x = np.array(df.X)
y = np.array(df.Y)

def best_fit_slop(xs,ys):
	m = (mean(xs)*mean(ys) - mean(xs*ys))/( mean(xs)**2 - mean(xs**2))
	return m

def y_intercept(xs,ys,m):
	b = mean(ys) - m*mean(xs)
	return b

m = best_fit_slop(x,y)
b = y_intercept(x,y,m)

regression_line = [(m*xx)+b for xx in x]

plt.scatter(df.X,df.Y)
plt.plot(x,regression_line)
plt.show()

plt.show()

#What Is Goodness-of-Fit for a Linear Model?

#R-squared is a statistical measure of how close the data are to the fitted regression line.

#R-squared = Explained variation / Total variation

#R-squared is always between 0 and 100%:
	#   0% indicates that the model explains none of the variability of the response data around its mean.
	#   100% indicates that the model explains all the variability of the response data around its mean.

#if a model could explain 100% of the variance, the fitted values would always equal the observed values and, therefore, all the data points would fall on the fitted regression line.

#R^2 = 1-( squared_error( calculated y) / squared_error( mean of y))

def squared_error(ys_original , ys_line):
	return sum ((ys_line - ys_original )**2)

def r_square(ys_original,ys_line):
	y_mean_line = [mean(ys_original) for y in ys_original]
	return 1 - (squared_error(ys_original,ys_line) / squared_error(ys_original, y_mean_line)) 

R_squared = r_square(y,regression_line)
print R_squared
