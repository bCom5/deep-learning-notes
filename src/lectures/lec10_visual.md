# Visualizing Deep Networks

### Activation Maximization
![](https://i.imgur.com/bKFfWQy.png)

Generate a synthetic image I, normalize it L2 nrom, maxing a class. This is done through backpropogation. Class is one hot encoded `[0 0 1 0 0]`. Images are weird though.

### Deconv Approaches
![](https://i.imgur.com/L9W4AWf.png)

To make generating synthetic images better:
* backprop zeros out negative values in the forward pass
* decovnet zeros out negative gradients in the backward pass. Negative gradients are inhibitory activations (they were the wrong class)
* guided backprop does both, zeros out both

### Neural Style Transfer
![](https://i.imgur.com/xdURQhp.jpg)

Review this!