import sys, os
import random
import numpy as np
from test.test_winreg import test_data
from matplotlib import pyplot as plt

class NetSimplified(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def train(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.."""       
        #jerzy added to solve python 2 -> 3 issues
        test_data = list(test_data)
        training_data = list(training_data)
        
        
        if test_data: n_test = len(test_data)
        n = len(training_data)
        print('train (SGD).epochs=',epochs)
        print('train (SGD).n_test=',n_test)
        print('train (SGD).n=',n)
        print('train (SGD).mini_batch_size=',mini_batch_size)

        for k in range(50000): 
             
            self.update_mini_batch(random.sample(training_data, 10), eta)
            #if k % 5000 == 0 and test_data: print (self.evaluate(test_data), n_test)
              
            

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
#         print('nabla_b[0]', nabla_b[0])
#         print('self.biases[0]',self.biases[0].shape) self.biases[0] (30, 1)
#         print('self.weights[0]',self.weights[0].shape) self.weights[0] (30, 784)
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
            
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        
        # feedforward
        activation = x
        outputs = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        
#         print('actBeforeFF.',activations[0].shape,'x',len(activations))
#         
#         print('0.',self.weights[0].shape,'x',len(self.weights)) #(30, 784) x 2
#         print('0.',self.biases[0].shape,'x',len(self.biases)) #(30, 1) x 2
#         
#         print('0.',self.weights[1].shape,'x',len(self.weights)) #(30, 784) x 2
#         print('0.',self.biases[1].shape,'x',len(self.biases)) #(30, 1) x 2
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z) #output
            
            outputs.append(activation) #list of outputs
        
#         print('activation0',activations[0].shape)
#         print('activation1',activations[1].shape)
#         print('activation3',activations[2].shape)       
#         print('activation-1',activations[-1].shape)
#         print('activation-2',activations[-2].shape)
        
       
        
#         print('actAfterFF.',activations[0].shape,'x',len(activations)) #(784, 1) x 3
#         print('actAfterFF.',activations[1].shape,'x',len(activations)) #(784, 1) x 3
#         print('actAfterFF.',activations[2].shape,'x',len(activations)) #(784, 1) x 3
        
        
        #sys.exit()
        
        # backward pass
        #delta equals to derivative of error function (Err)
        #nabla_w is weight increments
        #nabla_w is a weight change 
        delta = errorf_deriv(outputs[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, outputs[-2].transpose())
        
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
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, outputs[-l-1].transpose())
            
#         print("---",mse(outputs[-1],y))
        return (nabla_b, nabla_w)

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
def errorf_deriv(output_activations, y):
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."""
    return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))