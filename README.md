
# neural-network-compression
Neural Network Compression

# Objective

# Approached Methods
We have started with these papers [Deep Compression](https://arxiv.org/abs/1510.00149) and  [SqueezeNet](https://arxiv.org/abs/1602.07360).In SqueezeNet, the main objective is to make changes in architecture to have model compression. Each convolutional layer will be replaced with a Fire Module. The basic block diagram of Fire Module is below:
<!---![Fire Module](https://github.com/prashantksharma/neural-network-compression/blob/master/fire_module.png ) -->
<img src="https://github.com/prashantksharma/neural-network-compression/blob/master/fire_module.png" width="400" height="400">

The strategy behind using this Fire Module is to reduce the size of kernels. The other two strategies used in Squeezenet are Decreasing the number channels and Downsampling late in the network to have larger activation maps.


# Papers
* squeezenet paper
* deep compression

## Experiment 
1. Cifar classification using pytroch squeezenet
2. MNIST using pytorch squeTezenet



## Deep Compression on MNIST LeNet-5
https://github.com/mightydeveloper/Deep-Compression-PyTorch


