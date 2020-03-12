# Generative Models

## Review

![](https://i.imgur.com/xRqRDYu.png)

## Generative Models
* Variational Auto-Encoder (VAE)
* Auto-Regressive Models
* Transformers
* Generative Adversarial Networks (next time)

## Generative Models

![](https://i.imgur.com/jsJTn7h.png)

* __Classifier__ vs __Generator__
    * Deep Learning can't be efficient to create full joint probability

## Auto-Encoder

![](https://i.imgur.com/0uw6wI7.png)

* __Encoder:__ Data Compressor
* __Code:__ low dimension representation with information bottleneck
* __Decoder:__ generative

![](https://i.imgur.com/dInFxyq.png)

* One option is to invert the function
    * z -> x' is non-differentiable so it requires reinforcement learning

### Implicit Auto-Encoders

![](https://i.imgur.com/tZgN5Iz.png)

* From Latent code `z` to output `x`, same \\( \theta \\)
    * VAE: approximation Instead you get a distribution, not a single `x`

### Variational Auto-Encoders

![](https://i.imgur.com/CByY1dP.png)

* \\( q_{\phi}(z|x) \\): going up approximate inverse
    * \\( p_{\theta}(z|x) \\)
* What are these equations?
    * Theta vs phi

![](https://i.imgur.com/cl6xJLd.png)

* Therefore there are two sets of parameters instead

![](https://i.imgur.com/N96QvIi.png)

* \\( \epsilon \\) is noise
    * g can be a normal network
    * Both parts can be learn by backprop

![](https://i.imgur.com/FAKPt7t.png)

* Max the probability of x given
    * marginal likelihood so we have un-marginalize it
    * written as an expectation (Expectation from \\( E_{z\~q_{\phi}(z|x^i)} \\))

![](https://i.imgur.com/llP1nTk.png)

* It's like the MLE trick

![](https://i.imgur.com/bQUYyf3.png)

* Since it's concave, there's a __tangent line__
    * always less than or equal to that point

![](https://i.imgur.com/jdAxioF.png)

![](https://i.imgur.com/dMF1SMB.png)

* ?

## Optimizing Reparametrized Models

![](https://i.imgur.com/MhtvPk9.png)

* 

![](https://i.imgur.com/3q6PAJ5.png)

* LMAO I'm confused plz help
* Expected value: Integral
* repraram the network with epsilon noise
* the final equation is like SGD?
    * I think they meant it's like sampling 

![](https://i.imgur.com/VdhyTvM.png)

* We can approximate things

![](https://i.imgur.com/I9m3zRo.png)

* Let's be honest this is black magic at this point.
    * High variance from samples?
    * sample from equiv distributions instead
    * Wait this actually makes sense

![](https://i.imgur.com/d1vWYxe.png)

![](https://i.imgur.com/HsV8Cey.png)

* Can kind of generate faces

## Autoregressive Models

![](https://i.imgur.com/GuEsiT8.png)

* Pixels are correlated
* \\( p(x_i | x_{i-1}, x_{i-2}, ...) \\) conditioned on previous inputs

![](https://i.imgur.com/vtmFje2.png)

![](https://i.imgur.com/1MrQGev.png)

* CNN is an empty image

![](https://i.imgur.com/tYVH9Ia.png)

* and feed it back in

![](https://i.imgur.com/jg7dJ2K.png)

* slow generation, fast training! it can be done in parallel!

![](https://i.imgur.com/E0U6ytC.png)

* Use LSTM from row above, therefore can generate row in parallel!

![](https://i.imgur.com/E0U6ytC.png)

* bam!

![](https://i.imgur.com/adhQ1Xj.png)

* You can use previous rows, or all seen before!

### Examples

![](https://i.imgur.com/6TbDfEO.jpg)

* Good but with distortions

![](https://i.imgur.com/RbhuYF5.jpg)

* Ok

## Image Transformer

![](https://i.imgur.com/WJOARPJ.png)

* Like generating text
    * multi-head attention

![](https://i.imgur.com/f5LrKTS.png)

* 2D Memory Block, left is a transformer