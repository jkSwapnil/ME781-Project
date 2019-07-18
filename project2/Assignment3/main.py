import matplotlib.pyplot as plt
import numpy as np
import math

def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x)
    return exps / np.sum(exps)

data_file="a3-nnet-training-data.csv"
with open(data_file, 'r') as df:
    data = df.readlines()
data = data[1:]
data = np.array([[float(col) for col in row.split(',')] for row in data])
np.random.shuffle(data)
data.reshape(12,3)
X=data[:,0:2]
X=np.transpose(X)
W=np.random.rand(2,4)
B=np.random.rand(4,1)
V=np.random.rand(4,3)
C=np.random.rand(3,1)
Y=np.random.rand(3,1)
T=np.array([3, 3, 1, 2, 3, 2, 1, 3, 1, 2, 2, 1]).reshape(12,1)
i=0
while(i<100):
	Z=np.matmul(np.transpose(W),X)+B
	Z=1 / (1 + np.exp(-Z)) # imlimentation of the sigmoid function
	Y=softmax(np.matmul(np.transpose(V),Z)+C)
	i+=1