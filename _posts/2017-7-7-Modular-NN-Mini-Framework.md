---
layout: post
title: Building a Modular Neural Network Mini-Framework
permalink: /blog/Modular-NN-Mini-Framework
---
My [first Medium post](https://medium.com/@shubhang.desai/a-soft-introduction-to-neural-networks-6986b5e3a127) introduced the concept of neural networks at a fairly high level. Now, as we dive deeper into neural networks, it is necessary that we create a more robust framework for experimentation and understanding. In this short tutorial, we will develop a modular micro-framework that we will leverage and continuously develop as we journey into more complex neural network architectures. Going through this exercise will also put you in a great state of mind to explore more into popular NN frameworks such as PyTorch or Keras.

## Independent Nodes
As you know, a neural network is a set of layers which are constituted of mathematical computations. Each of these layers are independent of each other, aside from the fact that the output of one is the input of another. As long as we create a node that can take any input and gives an output, we can stack them to create a neural network.

The same can be said about the backwards pass of a neural network. During backpropagation, we recurrently apply the chain rule to send gradients down the graph to the parameters. This means that, so long as each node's backwards pass can take the incoming gradient as input and output its internally computed gradient, we can stack them together and have a mathematically sound backwards pass.

The reason why we want to do this is so that we can create \"building blocks\" that we can use to build more complex networks. Instead of hard-coding each layer's computations, we can define a class of layer that can be used in a modular fashion.

## Localized Layers
Let's start by defining a class prototype of a neural network layer. Each layer has three stateful properties: its parameters, its cache, and the gradients computed on its parameters on the backwards pass. Each layer also has two behaviors: its forward pass which takes as input the output from the previous layer, and its backwards pass which takes as input the incoming gradients from the previous layer. With all of that in mind, here is the code for our Layer class prototype:
```python
class Layer(object):
    def __init__(self):
        super(Layer, self).__init__()
        self.params = {}
        self.cache = {}
        self.grads = {}

    def forward(self, input):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError
```

Using this object as its superclass, let's create a linear network layer. As we know, a linear layer's parameters are its weight matrix and its bias vector. We can initialize these parameters in the class initializer:
```python
class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super(LinearLayer, self).__init__()
        self.params["W"] = np.random.randn(input_dim, output_dim) * 0.01
        self.params["b"] = np.zeros(output_dim)
```
The forward pass simply multiplies the layer's input with its weight matrix and adds onto the result the bias vector. However, we also must think ahead to the backwards pass: we need the forward pass input to compute the gradients in the backwards pass, so we'll save the input to the cache. Here's the code for the forward pass:
```python
    def forward(self, input):
            output = np.matmul(input, self.params["W"]) + self.params["b"]
            self.cache["input"] = input
            return output
```
Finally, the backwards pass. The gradient on the bias vector is simply the incoming gradient with a dimension collapsed, and the gradient on the weight matrix is the matrix multiplication between the forward pass input and the incoming gradient (with some dimension-matching gymnastics involved). Here's the code for the backward pass:
```python
    def backward(self, dout):
            input = self.cache["input"]
            self.grads["W"] = np.matmul(input.T, dout)
            self.grads["b"] = np.sum(dout, axis=0)

            dout = np.matmul(dout, self.params["W"].T)
            return dout
```

## Localized Loss
The loss has the same behaviors as a layer (a forward and backwards pass), but since it has no parameters it also has no gradients (we do need to have a cache). This means that the loss prototype will be simpler:
```python
class Loss(object):
    def __init__(self):
        super(Loss, self).__init__()
        self.cache = {}

    def forward(self, input, y):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
```
Now let's create a Softmax-Cross-Entropy loss function that inherits from this super class:
```python
class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self):
        super(SoftmaxCrossEntropyLoss, self).__init__()

    def forward(self, input, y):
        batch_size = input.shape[1]
        indeces = np.arange(batch_size)

        exp = np.exp(input)
        norm = (exp.T / np.sum(exp, axis=1)).T
        self.cache["norm"], self.cache["y"], self.cache["indeces"] = norm, y, indeces

        losses = -np.log(norm[indeces, y])
        return np.sum(losses) / batch_size

    def backward(self):
        norm, y, indeces = self.cache["norm"], self.cache["y"], self.cache["indeces"]
        dloss = norm
        dloss[indeces, y] -= 1
        return dloss
```

## Modular Network
Now that we have some building blocks made, we need to create the network frame that will accept and execute these localized nodes. The network should accept a list of Layers, a Loss function, and any additional hyperparameters (we will restrict it to learning rate for now). Here's the initializer for the network:
```python
class Network(object):
    def __init__(self, layers, loss, lr):
        super(Network, self).__init__()
        self.layers = layers
        self.loss = loss
        self.lr = lr
```
The network has two main tasks: training and evaluation. During training, the network accepts a training batch of data and its targets, performs a forward pass over all its layers, computes the loss, backpropagates back into its layers' parameters, and performs a gradient update. We employ two loops to do this. The first loops through the Layers list for the forward pass, and the second loops through the backwards Layers list for the backwards pass. We have a nested loop in the second one which performs gradient updates. In between the two main loops we have the Loss forward and backwards pass. We return the network's current predictions as well as its current loss value:
```python
def train(self, input, target):
        layers = self.layers
        loss = self.loss

        for layer in layers:
            input = layer.forward(input)

        l = loss.forward(input, target)
        dout = loss.backward()

        for layer in reversed(layers):
            dout = layer.backward(dout)

            for param, grad in layer.grads.items():
                layer.params[param] -= self.lr * grad

        return np.argmax(input, axis=1), l
```
During evaluation, the network is simply given data to do a prediction on. So, all that's necessary is a forward pass through all of its layers:
```python
def eval(self, input):
        layers = self.layers

        for layer in layers:
            input = layer.forward(input)

        return np.argmax(input, axis=1)
```
And we're doing with our Network object! This framework should work for **any** sequential neural network.

## Putting it All Together
Let's demonstrate this framework's functionality by replicating the simply, 1-hidden layer neural network we created in my previous post. First the boring stuff--importing packages, creating the data, and defining our accuracy function:
```python
from Layer import *
from Loss import *
from Network import *

X = np.array([[1, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0],
              [0, 1, 1, 0], [1, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 0],
              [0, 0, 1, 1], [1, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0],
              [1, 1, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
y = np.array([[0], [0], [0], [1], [1], [1], [0], [1], [1], [0], [1], [0], [1], [1], [0], [0]])

X_train = X[:8, :]
X_val = X[8:12, :]
X_test = X[12:16, :]
y_train = y[:8]
y_val = y[8:12]
y_test = y[12:16]

def eval_accuracy(output, target):
    pred = np.argmax(output, axis=1)
    target = np.reshape(target, (target.shape[0]))
    correct = np.sum(pred == target)
    accuracy = correct / pred.shape[0] * 100
    return accuracy
```
Now things get interesting. We define a list of Layers which contains one linear layer of input size 4 and output size 2, and we define a variable which is a SoftmaxCrossEntropyLoss. We use these variables to create a linear network:
```python
layers = [LinearLayer(4, 2)]
loss = SoftmaxCrossEntropyLoss()

linear_network = Network(layers, loss, 1e-2)
```
Finally, we train it!
```python
for i in range(4000):
    indeces = np.random.choice(X_train.shape[0], 4)
    batch = X_train[indeces, :]
    target = y_train[indeces]

    pred, loss = linear_network.train(batch, target)

    if (i+1) % 100 == 0:
        accuracy = eval_accuracy(pred, target)
        print("Training Accuracy: %f" % accuracy)

    if (i+1) % 500 == 0:
        accuracy = eval_accuracy(linear_network.eval(X_val), y_val)
        print("Training Accuracy: %f" % accuracy)

accuracy = eval_accuracy(linear_network.eval(X_test), y_test)
print("Test Accuracy: %f" % accuracy)
```
**You can check out the final code for this whole project [here](https://github.com/ShubhangDesai/nn-micro-framework).**

We're done! We've used our super cool neural network micro-framework to create a simple neural network much more easily than before. As you can see, this framework will be a great resource for learning and experimentation.

## Next Steps
As I said above, this exercise is a good way to transition into learning about frameworks like PyTorch and Keras. Both of these libraries have localized layers that can be stacked together in a modular network construct. The cool part is, we only have to directly code the forward pass; the backwards pass is done behind the scenes automatically. In PyTorch, it's even possible to create your own layers by defining their localized forward and backwards behaviors, making it even more similar to our mini-framework. Look out for the Sequential object in [these PyTorch examples](https://github.com/vinhkhuc/PyTorch-Mini-Tutorials) and [these Keras examples](https://keras.io/getting-started/sequential-model-guide/#examples) to see how these frameworks implement a modular network scheme.

Next week, I'll be continuing this teaching series by using our mini-framework to create a Convolutional Neural Network. It should be a ton of fun, so check back to read it! Happy learning!
