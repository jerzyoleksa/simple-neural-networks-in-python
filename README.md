# Neural networks handwritten letters recognition

[source]  

www: http://neuralnetworksanddeeplearning.com/chap1.html  
git: https://github.com/mnielsen/neural-networks-and-deep-learning.git

[changes]
- Refactored Python 2 to 3
- Added script for printing results and showing matlab graph
- Simplified code and its logical structure  

<h1>Michael Nielsen's famous code explanation</h1>
<p align="center">
  <img src="https://github.com/jerzyoleksa/simple-neural-networks-in-python/blob/master/images/nn2.png">
</p>

- 28x28 images from mnist make 784 input neurons, each for every pixel of the image
- 1st layer(input) consists of 784 neurons, 2nd layer(hidden) of 30 neurons and 3rd layer(output) of 10 neurons
- activations are outputs of layers, at the end of feed forward phase activations is a 3 element list of 784, 30 and 10 sized columns
