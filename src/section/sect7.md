# Section 7: Fooling Networks and Generative Models

## Deep Generative Model
* What is a deep generative model?
    * They represent the __full joint distribution__ \\( p(x, y) \\) where \\( x \\) is an input sample and \\( y \\) is an output or label
    * Review: `what is x (input sample)`?
* What types of deep generative models are there?
    * PixelRNN/PixelCNN (autogrgressive models), Variational Autoencoders (VAEe) and Generative Adversarial Networks (GANs)

## Autoregressive Models
* What is an __autoregressive model__?
    * a model that generates an image one pixel at a time
    * The probability of an image \\( x\\) is $$ p(x)=p(x_1,...,x_{n^2})=\prod_{i=1}^{n^2}p(x_i | x_1, ..., x_{i - 1}) $$ 
        * joint probability of all pixels, each pixel is the conditional probability of all past pixels
    * We model \\( p(x_i | x_1, ..., x_{i - 1}) \\) with a neural network \\( f_\theta \\)
    * Review: `how do we train f_theta`
* What is PixelCNN?
    * Uses a CNN to model the probability of a pixel given previous pixels.
* What are tradeoffs of PixelCNN?
    * It is slow at generating images. It has to go through each pixel, single pass through the image.
    * It is fast to train. It only has to go through a single pass through the image
    * Review: `PixelCNN`
* What is PixelRNN?
    * Uses a RNN (LSTM) to model the probability of a pixel given previous pixels
    * Remembers distant pixels
* Tradeoffs of PixelRNN?
    * It is fast at generating images, can generate full rows in parallel
    * It is slow at training, each row has a hidden state (?)
    * Review: `PixelRNN`
* What are the labels?
    * The image of a class is given. Try to predict the next pixel
* What is a Image Transformer?
    * Like PixelCNN but uses a __multi-headed attention network__

## Variational Autoencoders
* What is an __autoencoder__?
    * It's a model given data \\( x \\), predict \\( x \\) while compressing it into a smaller hidden state \\( z \\)
* What is an __encoder__?
    * An encoder takes a high-dimensional input \\( x \\) and outputs parameters of a Gaussian distribution (mean and variance \\( \mu_{z|x} \\) and \\( \Sigma_{z|x} \\)) that specify hidden variable \\( z \\).
    * We can model this with a deep neural network \\( q_\phi(z|x) \\)
* What is a __decoder__?
    * The decoder outputs a gaussian \\( \mu_{x|z} \\) and \\( \Sigma_{x|z} \\)) which we sample from to get \\( \hat{x} \\)
    * We can model this with a deep neural network \\( p_\theta(x|z) \\)
* How does a VAE generate images?
    * It samples latent variable \\( z \\) then samples \\( \hat{x} \\) from \\( z \\)
* How do you train a VAE?
    * Find parameters that maximize the likelihood of the data
    * This is intractable because of \\( z \\)

$$ p_\theta(x)=p_\theta(x_1,...,x_{n^2})=\int p_\theta(z)p_\theta(x|z)dz $$

![](https://i.imgur.com/o5MOlGZ.png)

* What is the __reparameterization trick__?
    * (?) look at lecture 17 slides
    * lots of complicated math
    * Review `VAE`

![](https://i.imgur.com/N96QvIi.png)

![](https://i.imgur.com/llP1nTk.png)

* How do you sample from the Expectation?
* Jensen's Inequailty

![](https://i.imgur.com/VdhyTvM.png)

* Just solve this instead

## Generative Adversarial Networks
* What is the difference between a GAN and a Autoregressive Model or a VAE 
    * Autoregressive Models model an explicit tractable density
    * VAE model an explicit intractable density
    * GANs don't model a density. They model it implicity in a two-player game with a generator and discriminator (what the heck?)
    * Review: `GAN`
* What is a __generator__?
    * \\( G \\) tries to generate real looking images \\( x \\) of input class given noise \\( z \\) (Gaussian or uniform distribution)
* What is a __discriminator__?
    * D tries to classify fake images.
* What is the GAN game?
    * \\( \theta_G \\) and \\( \theta_D \\) are parameters

![](https://i.imgur.com/6LuVsQs.png)

* What is a trick used for better gradient signals?
    * instead of minimizing, maximize below. the gradients are more stable

![](https://i.imgur.com/0nOvDQp.png)

## Other Questions
* What is KL Divergence? $$ D_{KL}(p||q)=\sum_{i=1}^N p(x_i) \log(\frac{p(x_i)}{q(x_i)}) $$
    * \\( q(x) \\) approximates \\( p(x) \\). If they match divergence is zero, but can go up to infinity
