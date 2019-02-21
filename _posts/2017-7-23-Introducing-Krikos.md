---
layout: post
title: Introducing Krikos - A Python ML Framework for Learning and Experimentation
permalink: /blog/Krikos
---
I am pleased to announce that I have published my first Python library: Krikos!

If you have been reading my [previous posts](/blog/Modular-NN-Mini-Framework), you have followed along with me as we developed a neural network micro-framework, which I have been using to augment my tutorial series. I realized that this was a super cool project which I wanted to make accessible to anyone by open-sourcing it and making it available on PyPI. Now, anyone can contribute to Krikos, and anyone can use it to start learning about and experimenting with neural networks.

As a slimmed down neural network framework, it is perfect for the learning AI researcher: it is simple enough to pick up quickly, but barebones enough that you are heavily involved in the development of your NN. Krikos is easy to use and experiment with, but demands the programmer's effort and involvement in development.

## Installing Krikos
Setting Krikos up for use in your project is as easy as:
```
pip install krikos
```
That's it!

## Using Krikos to Learn about ML
Krikos currently has two primary packages: ```nn``` and ```data```.

### nn Package
There are currently four classes in ```nn```: ```Layer```, ```Loss```, ```Regularization```, and ```Network```. The ```Layer``` class defines various network layers; the ```Loss``` class defines losses which can be used as your network's objective; the ```Regularization``` class defines regularization that your network can employ; and the ```Network``` class defines an actual network architecture.

The superclasses can be inherited from to create your own layers, losses, and network architectures. For example, if you'd like to define a layer, simply do:
```python
from krikos.nn.layer import Layer

class CustomLayer(Layer):
  ...
```
The convention is to have a dictionary of parameters and gradients. Use parameters from the dictionary in the forward pass and compute gradients and save it to the dictionary in the backward pass. Also, a cache is used to save necessary values computed in the forward pass for the backward pass computation. The Loss requires only a forward and backward pass, as per convention. ***Following these conventions will allow your Layer to work with the Sequential network.***

The Network superclass is perhaps the most salient class to inherit from. It has a train and eval function, and it instantiates the list of classes with different forward pass depending on train-time or test-time. **To create your custom architecture, you can must call both forward and backward on all of its layers, and you must aggregate the gradients on your own.** In future releases, gradients may be aggregated automatically, and the programmer need only reset the gradients to zero. Any changes made to the scheme of gradient computation will be documented on this blog. More detailed documentation is coming very soon.

### data Package
The data class has the Loader superclass, which is used to create the CIFARLoader class. You can inherit from this class to create your own data loaders. Because the function to get a batch of data is already written, you must only take care of loading and preprocessing the data as you'd like.

There are also some useful functions in utils.py in the data package.

### Examples
You can find examples of framework usage[here](https://github.com/ShubhangDesai/krikos/tree/master/examples). There are currently examples of fully-connected and convolutional networks.

## Contributing to Krikos
The source repository for Krikos can be found [here](https://github.com/ShubhangDesai/krikos/). Feel free to add layers, network architectures, etc. and submit pull requests. I am excited to see how this project is received and built upon!
