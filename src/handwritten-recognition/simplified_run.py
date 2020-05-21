import mnist_loader
import simplified
from simplified import NetSimplified

training_data, validation_data, test_data = mnist_loader.load_data_wrapper();

#jerzy added to solve python 2 -> 3 issues
test_data = list(test_data)
training_data = list(training_data)  

net = simplified.NetSimplified([784,30,10])


net.train(training_data, test_data=test_data)

test_data = list(test_data)
print('after training the list size should be 0:',test_data.__len__())

import numpy as np
imgnr = np.random.randint(0,10000)
imgnr = 7320;

#need to reload the test_data again, becauuse training empties the test_data list
training_data, validation_data, test_data = mnist_loader.load_data_wrapper();
test_data = list(test_data)
print(test_data.__len__())
#print('1.', test_data[imgnr])

prediction, zs = net.feedforward( test_data[imgnr][0] ) #0 is the first prpperty of tuple and is numpy.ndarray

#print("selected image daata:", test_data[imgnr][0])
print("Image number {0} is a {1}, and the network predicted a {2}".format(imgnr, test_data[imgnr][1], np.argmax(prediction[-1])))
#print("reshaped:", np.reshape(test_data[imgnr][0], (28,28) ))

#Image number 4709 is a 2, and the network predicted a 2
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2,figsize=(8,4))
ax[0].matshow( np.reshape(test_data[imgnr][0], (28,28) ), cmap='gray' )
ax[1].plot( prediction[-1], lw=3 )
ax[1].set_aspect(9)
plt.show()    