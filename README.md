## Neural network from scratch in python without any framework.

Neural Networks as a mathematical function that maps a given input to a desired output.

Neural Networks consist of the following components

1. An input layer, x
2. An arbitrary amount of hidden layers
3. An output layer, ŷ
4. A set of weights and biases between each layer, W and b
5. A choice of activation function for each hidden layer, σ. We’ll use a Sigmoid activation function.

![alt text](https://github.com/dilipajm/neural-network-python/blob/master/img/1.png)

### Training the Neural Network

The output ŷ of a simple 2-layer Neural Network is:

![alt text](https://github.com/dilipajm/neural-network-python/blob/master/img/2.png)

You might notice that in the equation above, the weights W and the biases bare the only variables that affects the output ŷ.

Naturally, the right values for the weights and biases determines the strength of the predictions. The process of fine-tuning the weights and biases from the input data is known as training the Neural Network.

Each iteration of the training process consists of the following steps:
Calculating the predicted output ŷ, known as feedforward
Updating the weights and biases, known as backpropagation

The sequential graph below illustrates the process:

![alt text](https://github.com/dilipajm/neural-network-python/blob/master/img/3.png)

### Feedforward
As we’ve seen in the sequential graph above, feedforward is just simple calculus and for a basic 2-layer neural network, the output of the Neural Network is:

![alt text](https://github.com/dilipajm/neural-network-python/blob/master/img/4.png)

### Loss Function
There are many available loss functions, and the nature of our problem should dictate our choice of loss function. In this tutorial, we’ll use a simple sum-of-sqaures error as our loss function.

![alt text](https://github.com/dilipajm/neural-network-python/blob/master/img/5.png)

That is, the sum-of-squares error is simply the sum of the difference between each predicted value and the actual value. The difference is squared so that we measure the absolute value of the difference.

Our goal in training is to find the best set of weights and biases that minimizes the loss function.

### Backpropagation
Now that we’ve measured the error of our prediction (loss), we need to find a way to propagate the error back, and to update our weights and biases.

In order to know the appropriate amount to adjust the weights and biases by, we need to know the derivative of the loss function with respect to the weights and biases.

Recall from calculus that the derivative of a function is simply the slope of the function.

If we have the derivative, we can simply update the weights and biases by increasing/reducing with it(refer to the diagram above). This is known as gradient descent.

However, we can’t directly calculate the derivative of the loss function with respect to the weights and biases because the equation of the loss function does not contain the weights and biases. Therefore, we need the chain rule to help us calculate it.

![alt text](https://github.com/dilipajm/neural-network-python/blob/master/img/6.png)

Chain rule for calculating derivative of the loss function with respect to the weights. Note that for simplicity, we have only displayed the partial derivative assuming a 1-layer Neural Network.

The derivative (slope) of the loss function with respect to the weights, so that we can adjust the weights accordingly.




