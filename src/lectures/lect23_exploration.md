# Exploration

![](https://i.imgur.com/gVCuOe3.png)

* Q-functions expected value of rew
* bottom term is the policy gradient
* Advantage Functions
    * how much better it is

![](https://i.imgur.com/MQJYkvf.png)

![](https://i.imgur.com/lFJtqCC.png)

* Actor-Critic Function?

![](https://i.imgur.com/kQj1Usk.png)

* use a replace buffer to avoid bias
* use a target network

![](https://i.imgur.com/XjLzegW.png)

* two q functions

## Exploration
* optimistic exploration
* posterior sampling
* curiosity-driven exploration (model-based)
* Information Bottleneck Methods

## What is the Problem?

![](https://i.imgur.com/PrlBsgu.png)

* Policy
* Value-Based, you actually have to visit the state
* Epsilon greedy sometimes choose random action

![](https://i.imgur.com/ITiJxXx.png)

* random walk explores slowly

![](https://i.imgur.com/Wz79mAd.png)

* Montezuma's Revenge is impossible
    * hard to randomly get key, avoid skull and get to the exit

## Exploration vs Exploitation
    * try new resturant
    * go to best restaurant

## Optimistic Exploration in RL

![](https://i.imgur.com/U4JLoOe.png)

* __Upper Confidence Bount__ (UCB)
    * num times we have visited state s
    * intrinsic reward for bonus

## Similar States?

![](https://i.imgur.com/B0ZOYFl.png)

* density of a state
* updating the counts

## Exploring with Pseudo-counts

![](https://i.imgur.com/IukNx9j.png)

* pseudo count now have a n and N

![](https://i.imgur.com/iBrD3CF.png)

* It's a generative model
* PixelCNN is pretty good!

## What kind of bonus to use?

![](https://i.imgur.com/KOmepsX.png)

* UCB is good

## Posterior Sampling

![](https://i.imgur.com/7w56Kuw.png)

* uncertainty?
    * sample parameters (Thompson Sampling)

## Bootstrap

![](https://i.imgur.com/7hif03q.png)

* resample with replacement D_N samples
    * train model on each sample
    * to "sample" evaluate one of the models
* you can explore the parameter space with bootstrap samples
* a trick is multihead learning, share the convolution layers

![](https://i.imgur.com/Pryjykv.png)

* consistent for an entire episode (COMMIT TO THE CHOICE)

## Model Free RL

![](https://i.imgur.com/JhsdfT4.png)

* don't care about reward and transition

## Model-based RL

![](https://i.imgur.com/ddSKt3U.png)

* It helps!
* Learn a latent space \\( \phi(s) \\)

## Curiosity-driven

![](https://i.imgur.com/TAgyxSa.png)

## Intrinsic Curiosity Model (ICM)

![](https://i.imgur.com/1iqIc8n.png)

* useful embedding
* latent space to learn an inverse function

![](https://i.imgur.com/XM7wFYb.png)

* ?

![](https://i.imgur.com/sYEyQTN.png)

* works really well!

## Information Bottleneck Approaches

![](https://i.imgur.com/qMoalV5.png)

* make it as minimal as possible

## Mutal Information


* __Mutual Information__: joint distribution?

![](https://i.imgur.com/LGzajmO.png)

![](https://i.imgur.com/LAlolmh.png)

* Latent layer Z, it's a distribution but in a network it's deterministic

![](https://i.imgur.com/teuEH1l.png)

* want to minimize redundant info

![](https://i.imgur.com/V0CvkyB.png)

* better generalization

![](https://i.imgur.com/i1k7ut9.png)

## Information Bottleneck for RL

![](https://i.imgur.com/YS1LKvh.png)

* subpolicy

![](https://i.imgur.com/zQVqSRs.png)

* Learns to move on the skeleton (vertices)
* from infinite number of decisions to a small finite number of them

## Adversarial Information Bottlenecks

![](https://i.imgur.com/ciql9m7.png)

* train mutual info and detect the bottleneck

![](https://i.imgur.com/Fj24lYe.png)

* rip

![](https://i.imgur.com/RGeKtMV.png)

* learn where you don't need to take actions but eventually states with high I

## Questions
* What to review
