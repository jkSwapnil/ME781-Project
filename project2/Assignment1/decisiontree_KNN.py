import argparse
from sklearn import neighbors
from sklearn import tree
import numpy as np

data_file="4b.csv"
with open(data_file, 'r') as df:
    data = df.readlines()

data = data[1:]
data = np.array([[float(col) for col in row.split(',')] for row in data])
np.random.shuffle(data)

train_x= data[:,0]
train_y= data[:,1]
train_x=train_x.reshape(-1,1)
predict_x=np.array([13.3375154901, -0.9624957759, 6.0984510789, -2.2003260581, 14.0995777678, 0.5029836949, -1.0622723773, -4.5455076639, 13.3825418027, -4.2196210148, 13.7887296453, -0.7769178925, 9.2603615159, -1.2492567534, -0.31135906])
#Decision tree
predict_x=predict_x.reshape(-1,1)
print("From Decision trees")
clf = tree.DecisionTreeRegressor(random_state=0, min_samples_leaf=5)
clf.fit(train_x, train_y)
predictions1 = clf.predict(predict_x)
print(predictions1)
print("From KNN")

knn = neighbors.KNeighborsRegressor(3, weights='uniform')
predictions2=knn.fit(train_x, train_y).predict(predict_x)
print(predictions2)