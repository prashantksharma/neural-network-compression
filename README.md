
# neural-network-compression
Neural Network Compression

# Objective
In order to deploy deep learning models on to low power devices or low memory devices or both, there needs to reduction in size of models. This reduction is basically done by both using architectural changes and using techniques like Pruning, Huffman Coding and Weight sharing.
# Approached Methods
We have started with these papers [Deep Compression](https://arxiv.org/abs/1510.00149) and  [SqueezeNet](https://arxiv.org/abs/1602.07360).In SqueezeNet, the main objective is to make changes in architecture to have model compression(reduction in number of parameters used) without significant loss in accuracy. Each convolutional layer will be replaced with a Fire Module. The basic block diagram of Fire Module is below:
<!---![Fire Module](https://github.com/prashantksharma/neural-network-compression/blob/master/fire_module.png ) -->
<img src="https://github.com/prashantksharma/neural-network-compression/blob/master/fire_module.png" width="400" height="400">

The strategy behind using this Fire Module is to reduce the size of kernels. Fire module consists of two layers: 
* Squeeze Layer
* Expand Layer.\\
In the figure explained above, squeeze layer have only **1x1** filters and Expand layer consists both **1x1 and 3x3** filters. So, there are three tunable parameters in this Fire Module, i.e sizes of 1x1 in squeeze layer and sizes of 1x1 and 3x3 in expand layer. The other two strategies used in Squeezenet are Decreasing the number channels and Downsampling later deep in the network to have larger activation maps.


# Papers
* squeezenet paper
* deep compression

## Experiment 
1. Cifar classification using pytroch squeezenet
2. MNIST using pytorch squeTezenet



## Deep Compression on MNIST LeNet-5
https://github.com/mightydeveloper/Deep-Compression-PyTorch


