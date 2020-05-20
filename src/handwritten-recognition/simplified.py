import sys, os
import random
import numpy as np
from test.test_winreg import test_data
from matplotlib import pyplot as plt
import pickle

class NetSimplified(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        
#         self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
#         self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
#         pickle_out = open("dict.pickle2","wb")
#         pickle.dump(self.biases, pickle_out)
#         pickle_out.close()


        #load serialized objects to avoid random values
        pickle_in = open("dict.pickle","rb")
        self.weights = pickle.load(pickle_in)        
        pickle_in = open("dict.pickle2","rb")
        self.biases = pickle.load(pickle_in)
       
        
       
        
#         print(self.biases);
#         print(self.weights);
        

        



    def train(self, training_data, test_data=None):
     
        #jerzy added to solve python 2 -> 3 issues
#         test_data = list(test_data)
#         training_data = list(training_data)
#         
#         
#         if test_data: n_test = len(test_data)
#         n = len(training_data)


        pickle_in = open("dict.pickle3","rb")
        samplearr = pickle.load(pickle_in) 
        
        
#         print(self.weights[0][10][4],'/END')
        
        for k in range(50000): 
             
            sample = samplearr[k]

            for x, y in sample:
                       # feedforward
                activations, zs = self.feedforward2(x, self.biases, self.weights)
                bias_change, weight_change = self.backprop(x, y, self.biases, self.weights, activations, zs)
                
                #should we move the bias and weight updating into backrop - YES, backprop role is to update weights
                self.biases = list(map(lambda x,y: x - 0.3*y, self.biases, bias_change))        
                self.weights = list(map(lambda x,y: x - 0.3*y, self.weights, weight_change))
        
            #if k % 5000 == 0 and test_data: print (self.evaluate(test_data), n_test)
    
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def feedforward2(self, x, biases, weights):
        activation = x
        activations = []
        activations.append(x) # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        
        for b, w in zip(biases, weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z) #output
            
            activations.append(activation) #list of activations
            
#         print('a-s:',activations[0].shape,activations[1].shape,activations[2].shape)
#         print('z-s:',zs[0].shape,zs[1].shape)    
        return (activations, zs)

    def backprop(self, x, y, biases, weights, activations, zs):

        bias_change = clone_empty_matrix( biases )
        weight_change = clone_empty_matrix( weights )



        # backprop
        delta = error_deriv(activations[-1], y) * sigmoid_deriv(zs[-1])
        bias_change[-1] = delta
        weight_change[-1] = np.dot(delta, activations[-2].transpose())
        
        #-1 means the last layer, the output layer ?
#         print('1.',delta.shape) #(10, 1)
#         print('2.',nabla_b[-1].shape) #2. (10, 1)
#         print('3.',nabla_w[-1].shape) #3. (10, 30)
#         plt.imshow(delta, interpolation='nearest')
#         plt.table(cellText=delta,loc='center',cellLoc='center')
#         plt.show()

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_deriv(z)
            delta = np.dot(weights[-l+1].transpose(), delta) * sp
            bias_change[-l] = delta
            weight_change[-l] = np.dot(delta, activations[-l-1].transpose())
            
#         print("---",mse(activations[-1],y))
        return (bias_change, weight_change)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

# def mse(outputs, expected):
#     return sum((y - t) ** 2 for (y, t) in zip(outputs, expected)) / len(outputs)
#     
def error_deriv(output_activations, y):
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."""
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