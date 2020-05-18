# Neural networks handwritten letters recognition

[source]  

www: http://neuralnetworksanddeeplearning.com/chap1.html  
git: https://github.com/mnielsen/neural-networks-and-deep-learning.git

[changes]
- Refactored Python 2 to 3
- Added script for printing results and showing matlab graph
- Simplified code and its logical structure  

<h1>Michael Nielsen's famous 74-liner explanation</h1>
<p align="center">
  <img src="https://github.com/jerzyoleksa/simple-neural-networks-in-python/blob/master/images/nn2.png">
</p>

- 28x28 images from mnist make 784 input neurons, each for every pixel of the image
- backprop function includes the feed forward phase which can be misleading
- 1st layer(input) consists of 784 neurons, 2nd layer(hidden) of 30 neurons and 3rd layer(output) of 10 neurons
- activations are outputs of layers, at the end of feed forward phase activations is a 3 element list of 784, 30 and 10 sized columns
- the last activation - the activation[2] - is not realy an activation, as the last layer, doesnt have any following layer and does not activate anything
- python lets you use negative array indexes to get the last element: activations[-1], or the second last: activations[-2]
- first column of activations variable is just an input - x, the next two are the calculated outputs
- in python tuple is just a pair of values, zip is a function that creates these pairs
- nabla word is an unnecessary complication, i would name it updated weights instead
- variable z is the weighted sum of neuron inputs plus one bias value
- why do we use MSE (Mean Squared Error) as a Loss Function ? Not only because we want to sum absolute values of errors, so they dont hide each other when having oposite signs, but also because we want errors to have similar values (rather than having some nodes with very small error and others with very large)
- and remember, later on after we calculate the Error function we will use gradient descent to minimize it
- as a java guy, i found "list comprehension" i.e. for loops embeded in square brackets - hard to read, so i will try to replace it with a simple, more universal notation
