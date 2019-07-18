import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

data_file="a2-data-set.csv"
#reading D1 and D2
with open(data_file, 'r') as df:
    data = df.readlines()
data = data[1:]
data = np.array([[float(col) for col in row.split(',')] for row in data])
np.random.shuffle(data)
D1=data[0:399,:]
D2= data[399:499,:]
#reading D3
data_file="a2-data-set-D3.csv"
with open(data_file, 'r') as df:
    D3 = df.readlines()
D3 = D3[1:]
D3 = np.array([[float(col) for col in row.split(',')] for row in D3])
np.random.shuffle(D3)
#reading D4
data_file="a2-data-set-D4.csv"
with open(data_file, 'r') as df:
    D4 = df.readlines()
D4 = D4[1:]
D4 = np.array([[float(col) for col in row.split(',')] for row in D4])
np.random.shuffle(D4)

clf = RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=0)
clf.fit(D1[:,0:1], D1[:,2])
print("ROOT_MEAN_SQUARED_ERROR")
predictions1 = clf.predict(D1[:,0:1])
print(sqrt(mean_squared_error(predictions1, D1[:,2])))
predictions2 = clf.predict(D2[:,0:1])
print(sqrt(mean_squared_error(predictions2, D2[:,2])))
predictions3 = clf.predict(D3[:,0:1])
print(sqrt(mean_squared_error(predictions3, D3[:,2])))
predictions4 = clf.predict(D4[:,0:1])
print(sqrt(mean_squared_error(predictions4, D4[:,2])))
print("MEAN_ABSOLUTE_ERROR")
predictions1 = clf.predict(D1[:,0:1])
print(mean_absolute_error(predictions1, D1[:,2]))
predictions2 = clf.predict(D2[:,0:1])
print(mean_absolute_error(predictions2, D2[:,2]))
predictions3 = clf.predict(D3[:,0:1])
print(mean_absolute_error(predictions3, D3[:,2]))
predictions4 = clf.predict(D4[:,0:1])
print(mean_absolute_error(predictions4, D4[:,2]))