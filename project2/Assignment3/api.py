import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.neural_network import MLPClassifier


data_file="sample.csv"
with open(data_file, 'r') as df:
    data = df.readlines()
data = data[1:]
data = np.array([[float(col) for col in row.split(',')] for row in data])
np.random.shuffle(data)

mlp= MLPClassifier(hidden_layer_sizes=(2))
mlp.fit(data[:,0:2],data[:,2])
print("results")
print(mlp.predict(data[:,0:2]))
print("coeficients")
print(mlp.coefs_)
print("intercepts")
print(mlp.intercepts_)