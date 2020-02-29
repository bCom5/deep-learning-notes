# Attention Networks

__The most important idea in deep networks this decade.__

### Pilot analogy
![](https://i.imgur.com/LU12T8h.jpg)

Pilots move their focus on different things, gauges, buttons, switches


### Papers
Attention is All You Need (2017) only stacked attention, previous used RNNs

## Soft vs Hard Attention Introduction
![](https://i.imgur.com/GwTuFpG.png)

* __Hard Attention__ is what humans do. Not differentiable because attention map is 1 where focused, 0 elsewhere (discrete).
* __Soft Attention__ - linear combination over inputs, can use backprop.
    * Trains network and attention to the right place!

![](https://i.imgur.com/3JN8VEn.png)
__Supervised Learning__ vs __Reinforcement Learning__, maximizing a reward (discrete)

Reinforcement Learning, the __agent tries to maximize the sum of rewards over an epoch__

## Attention for Recognition
![](https://i.imgur.com/d2tfdX3.png)

Has time stamps. Has location and the image. RNN always predicting gaze location for next time.

![](https://i.imgur.com/BAmjvmm.png)

Resolution when focusing on something (glimpse). Output is location and action (classify/don't classify). Inputs are on top, output is at the bottom.

![](https://i.imgur.com/NnoPB1G.png)

This shows glimpses. Focused in center. Blurred around. More blurred after, 3 levels. Green is location as it moves around.


## Soft Attention for Translation
![](https://i.imgur.com/ode4uKw.png)

When the world is outputting _Me_, it has attention on _I_. Likewise with _coffee_ and _cafe_.

![](https://i.imgur.com/bWX5rXq.png)

What word does each word correspond to? (Where is the attention located)?

## RNN for Captioning
![](https://i.imgur.com/O3QzfWP.png)

Output `d0` goes into a softmax over the vocabulary. _Bird_ is fed back into `x1`, next token. 

![](https://i.imgur.com/qA37EVl.png)

Have \\( L \times D \\) features extracted. Our RNN outputs classification and weight location. Mask/summing (?) with the features give the weighted feature \\( D \\) fed back into the RNN.

## Soft vs. Hard Attention in Captioning
![](https://i.imgur.com/VNdA0yR.png)

* Have two inputs, RNN gives probability distribution (see above) (also think about the result of softmax).
* __Soft Attention__ sum all location and derivative is \\( \frac{dz}{dp} \\). Train with gradient descent.
* __Hard Attention__ samples only one location. With argmax, \\( \frac{dz}{dp} \\) is almost zero everywhere. Can't use gradient descent.

![](https://i.imgur.com/RphCcoj.png)

Soft is smooth around the area. Hard is discrete (1 or 0) around the image.

![](https://i.imgur.com/wLcn23X.jpg)

![](https://i.imgur.com/YYT5G2u.jpg)

Can see where it made mistakes!!!

## Attention Mechanics
![](https://i.imgur.com/0q6PVTx.png)

* Image Feature output matrix \\( L \times D \\)
* Attention weights output vector \\( L \\). Can \\( L \\) be an image \\( H \times W \\)?
* Do a broadcast multiply - each vector in \\( D \\) multiplies by the Attention vector. Output is still \\( L \times D \\)
* Sum the \\( L \\) axis (vector). Output is size \\( D \\).

![](https://i.imgur.com/NcPqXPf.png)

* Need sum for the contributions. This is because \\( B \\) is broadcast multiplied to \\( A \\)

### Salience
![](https://i.imgur.com/v4cwBwH.png)

* Circle operation is product
* What is salience (A \*dot\* )
    * __Salience:__ The total contribution and gradient to the output (the importance of an area)

## Attention and LSTM
![](https://i.imgur.com/EDcAoA5.png)

* LSTMs also receive a __salience gradient__. There is multiplicative gating (kind of like attention)

## Soft Attention for Video
![](https://i.imgur.com/wHL1iwj.png)

Now temporal (time) can have attention. (Which frame is important?)

![](https://i.imgur.com/lo2rgUy.png)

* Can have spatial and temporal attention. Temporal in this network.
* Create a vector of feature frames.

![](https://i.imgur.com/uvLw16A.png)

![](https://i.imgur.com/iNBaAPT.png)

Perplexity is diversity of word options. Goal is smaller which means the model is more confident.

## Find the Region Again
![](https://i.imgur.com/gVzxWMc.png)

* Classify - attention to regions in input
* Generate - draw digit

Has a parameter for scale

## Takeaways
![](https://i.imgur.com/vquu8LF.png)

* __Salience:__ Emphasize imporant data.

![](https://i.imgur.com/ddEkm4X.png)