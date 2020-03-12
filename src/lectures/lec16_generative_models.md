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
