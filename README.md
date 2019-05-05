
# neural-network-compression
Neural Network Compression

# Objective
Recent research on deep convolutional neural networks (CNNs) has focused pri-marily on improving accuracy. For a given accuracy level, it is typically possible to identify multiple CNN architectures that achieve that accuracy level. With equivalent accuracy, smaller CNN architectures offer at least three advantages:
1. Require less communication across servers during distributed training.  
2. Require less bandwidth to export a new model to client over the cloud. 
3. More feasible to deploy on FPGAs and other low power devices or low memory devices.

This reduction in model size is basically done by both using architectural changes and using techniques like Pruning, Huffman Coding and Weight sharing.

# Approached Methods
We have started with these papers [Deep Compression](https://arxiv.org/abs/1510.00149) and  [SqueezeNet](https://arxiv.org/abs/1602.07360).
## SqueezeNet
The main objective is to make changes in architecture to have model compression(reduction in number of parameters used) without significant loss in accuracy. Each convolutional layer will be replaced with a Fire Module. The basic block diagram of Fire Module is below:
<!---![Fire Module](https://github.com/prashantksharma/neural-network-compression/blob/master/fire_module.png ) -->
<img src="https://github.com/prashantksharma/neural-network-compression/blob/master/fire_module.png" width="400" height="400">

The strategy behind using this Fire Module is to reduce the size of kernels. Fire module consists of two layers: 
* Squeeze Layer
* Expand Layer.\\
In the figure explained above, squeeze layer have only **1x1** filters and Expand layer consists both **1x1 and 3x3** filters. So, there are three tunable parameters in this Fire Module, i.e sizes of 1x1 in squeeze layer and sizes of 1x1 and 3x3 in expand layer. The other two strategies used in Squeezenet are Decreasing the number channels and Downsampling later deep in the network to have larger activation maps.


# Papers
* squeezenet paper
* deep compression

## Dataset 
1. CIFAR10
2. MNIST 
## Arcitecture
1. LeNet5
2. VGGNet



## Deep Compression on MNIST LeNet-5
https://github.com/mightydeveloper/Deep-Compression-PyTorch


