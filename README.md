Binary Image Classifier

# Neural Network from Scratch

This project demonstrates a simple feedforward neural network built using only NumPy. It includes two dense layers, ReLU and Softmax activations, and uses categorical cross-entropy as the loss function. A basic training loop performs forward and backward passes, followed by gradient descent to update weights.

## Key Components

* Dense (fully connected) layers
* ReLU activation for hidden layers
* Softmax activation for output layer
* Categorical cross-entropy loss
* Manual backpropagation and parameter updates

## Workflow

1. Input data is passed through the network to produce predictions.
2. The loss is calculated using the predicted and true labels.
3. Gradients are computed using backward propagation.
4. Weights and biases are updated using gradient descent.

## Requirements

* Python 3
* NumPy

