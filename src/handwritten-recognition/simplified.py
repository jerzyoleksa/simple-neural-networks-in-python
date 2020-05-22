import sys, os
import random
import numpy as np
from test.test_winreg import test_data
from matplotlib import pyplot as plt
import pickle

class NetSimplified(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
#         pickle_out = open("dict.pickle2","wb")
#         pickle.dump(self.biases, pickle_out)
#         pickle_out.close()


        #load serialized objects to avoid random values
#         self.weights = loadSer("dict.pickle","rb")
#         self.biases = loadSer("dict.pickle2","rb")
#         samplearr = loadSer("dict.pickle3","rb")
#         sample = samplearr[k]


    def train(self, training_data, test_data):

      for i, (x, y) in enumerate(training_data): 
            activations, zs = self.feedforward(x)
            self.backprop(x, y, self.biases, self.weights, activations, zs)
                  
            if i % 5000 == 0 and test_data: print (self.evaluate(test_data),'/', len(test_data))
    
    
    
    def feedforward(self, a):
    
        activations = [a] #init list with a as the first element
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a)+b
            a = sigmoid(z)
            
            zs.append(z)
            activations.append(a)
 
        return (activations, zs)



    def backprop(self, x, y, biases, weights, activations, zs):

        bias_change = clone_empty_matrix( biases )
        weight_change = clone_empty_matrix( weights )

        delta = error_deriv(activations[-1], y) * sigmoid_deriv(zs[-1])
        bias_change[-1] = delta
        weight_change[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_deriv(z)
            delta = np.dot(weights[-l+1].transpose(), delta) * sp
            bias_change[-l] = delta
            weight_change[-l] = np.dot(delta, activations[-l-1].transpose())
            
        self.biases = list(map(lambda x,y: x - 0.3*y, self.biases, bias_change))        
        self.weights = list(map(lambda x,y: x - 0.3*y, self.weights, weight_change))
        
        

    def evaluate(self, test_data):
        result = []
        for (x, y) in test_data:
            
            activations, zs = self.feedforward(x)
            result.append((np.argmax(activations[-1]), y))
                        
        return sum(int(x == y) for (x, y) in result)

# def mse(outputs, expected):
#     return sum((y - t) ** 2 for (y, t) in zip(outputs, expected)) / len(outputs)
#     
def loadSer(name, mode):
    pickle_in = open(name, mode)
    return pickle.load(pickle_in)

def error_deriv(output_activations, y):
    return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))

def clone_empty_matrix(arr):
    copy_matrix = []    
    for i in range(0, len(arr)):  
        copy_matrix.append(np.zeros(arr[i].shape))
    return copy_matrix