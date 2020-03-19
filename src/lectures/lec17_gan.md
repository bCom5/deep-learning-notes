# Generative Adversarial Networks

## Last Time

![](https://i.imgur.com/LCa14jA.png)

* network \\( p_{\theta} \\)


![](https://i.imgur.com/ADhTUfX.png)

* PixelCNN

![](https://i.imgur.com/jD6YkiT.png)

* Autoregressive conditioned on the input
* Posterior distribution
* Model human discriminating

## Problems of Variational Auto-Encoders

![](https://i.imgur.com/0ZfYPLQ.png)

* Compares densities in __latent z space__

![](https://i.imgur.com/hSX7tTl.png)

![](https://i.imgur.com/srG2XiI.png)

* Avoid approximations and density estimates

![](https://i.imgur.com/srG2XiI.png)

* Real and Synthetic Image distribution \\( p_{re}(x) \\)
* Has KL divergence and approximate expected values by sampling

![](https://i.imgur.com/VoRvTK5.png)

* Wait what?
* \\( d_{\phi} \\) is softmax - so it is a classifier

![](https://i.imgur.com/q3iiMMI.png)

* First equation is plus
    * We are minimizing the divergence
* Second equation is minus.
    * We are optimizing the discriminator

![](https://i.imgur.com/miQHrAw.png)

* Proof of classifier

![](https://i.imgur.com/G0tU2ga.png)

* Written as one formula
* Now we add a generator \\( g_{\theta} \\) being maximized

![](https://i.imgur.com/rwJe94j.png)

* Two player game

![](https://i.imgur.com/DOJaVTh.png)

* Generate from Latent space to image

![](https://i.imgur.com/SOxLluY.png)

* Classifies realness. 
* Maximizes (or minimize) divergence? what does that mean

![](https://i.imgur.com/zMuCciB.png)

* Can backprop all the way! End-to-end

![](https://i.imgur.com/GTWN3FI.png)

* 

![](https://i.imgur.com/Nn5x37g.png)

* Step 1 update discriminator
* Step 2 update generator

![](https://i.imgur.com/8x6Zh9J.png)

* Big gradients when outputs are bad.

![](https://i.imgur.com/jcBeurj.png)

* Does pretty well! (2014)

## GAN Improvements

[GAN Hacks](https://github.com/soumith/ganhacks)

![](https://i.imgur.com/TvpPXWB.png)

* Progressively going up in
* Fractional Stride Convolution

![](https://i.imgur.com/yivjQGN.png)

* Can do arithmetic on this! Man with glasses! without!

![](https://i.imgur.com/ME1p60T.png)

* Add a class label

![](https://i.imgur.com/ME1p60T.png)

* Text to Image synthesis to both generator and discriminator
    * `This flower has small round violet petals with a dark purple center.`

## GAN Problems

![](https://i.imgur.com/2OnYLJR.png)

* only learns to generate South Pole
* But then starts to generate Alice Springs
* Unstable

![](https://i.imgur.com/aXGncYz.png)

* Experience replay - old discriminator and generator.

![](https://i.imgur.com/0m2YB88.png)

* Wasserstein Distance

![](https://i.imgur.com/UIgAppt.png)

* Critic function diffs approximate the Wasserstein GAN

![](https://i.imgur.com/s780qW2.png)

* Provably avoids mode collapse - train to optimality

![](https://i.imgur.com/yyXoOeQ.png)

![](https://i.imgur.com/a55wcBI.png)

* Scale up, don't need progressive GAN

![](https://i.imgur.com/IvEgb8x.png)

* Upsampling - generator

## Evaluating GANs

![](https://i.imgur.com/oawobnd.png)

* Inception Score
* Using Inception v3 Network
* Multi-resolution convolutions
    * 1x1, 3x3, 5x5 blocks
    * like humans

![](https://i.imgur.com/OL9fRyo.png)

* Another improvement on evaluation

## Unpaired Conditional Image Generation

![](https://i.imgur.com/ZfgRSf0.png)

* not paired. Use a corresponding image and generate a synthetic image

![](https://i.imgur.com/ZWXiMn2.png)

* __CycleGAN__ create a cycle 2 generators and 2 discriminators
    * cycle-consistency loss L1 distance

![](https://i.imgur.com/ZWXiMn2.png)

* Really remarkable
* Compared to neural style
    * Neural style uses statistics
    * Harder to discriminate