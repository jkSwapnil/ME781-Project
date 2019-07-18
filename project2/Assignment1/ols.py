import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

data_file="3d.csv"
with open(data_file, 'r') as df:
	data = df.readlines()

data = data[1:]
data = np.array([[float(col) for col in row.split(',')] for row in data])
np.random.shuffle(data)
validation=data[80:99,1]
validation.reshape(-1,1)


model = sm.OLS(data[0:79,1], data[0:79,0],'y','x')
results=model.fit() 
#print(results.pvalues)
#print(sqrt(mean_squared_error(results.predict(data[80:99,0]), data[80:99,1])))
#print mean_absolute_error(results.predict(data[80:99,0]), data[80:99,1])
#print(results.params)
#print(results.summary())
predict=[13.3375154901, -0.9624957759, 6.0984510789, -2.2003260581, 14.0995777678, 0.5029836949, -1.0622723773, -4.5455076639, 13.3825418027, -4.2196210148, 13.7887296453, -0.7769178925, 9.2603615159, -1.2492567534, -0.31135906]
print(results.predict(predict))