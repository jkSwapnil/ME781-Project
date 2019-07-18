import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

data_file="no_NA.csv"
with open(data_file, 'r') as df:
	data = df.readlines()

data = data[1:]
data = np.array([[float(col) for col in row.split(',')] for row in data])
np.random.shuffle(data)

model = sm.OLS(data[0:59,1], data[0:59,0],'y','x')
results=model.fit() 
print(results.pvalues)
print(sqrt(mean_squared_error(results.predict(data[60:84,0]), data[60:84,1])))
print mean_absolute_error(results.predict(data[60:84,0]), data[60:84,1])
print(results.params)
print(results.summary())