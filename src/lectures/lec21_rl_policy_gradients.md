# Reinforcement Learning: Policy Gradients
[Link](https://www.youtube.com/watch?v=rpKkZ71A_xE)

![](https://i.imgur.com/1JuznSr.png)

* You have a:
    * __state__
    * which you observe with an __observation__
    * you choose what action to take (__policy__)
        * \\( \pi_{\theta} ( a_t | o_t ) \\)

![](https://i.imgur.com/2NCCfsM.png)

* we use a reward function
    * what is the best action for this state?

## Markov Decision

![](https://i.imgur.com/txTpesj.png)

* Markov Chain
* transition is indepedent of state before previous state

![](https://i.imgur.com/dTVyPYO.png)

* Transition is a tensor with state and action

![](https://i.imgur.com/txTpesj.png)

* reward function takes a state and action 
* trajectory is a sequence of states and actions

![](https://i.imgur.com/HEHm68j.png)

* states are orange
* actions are green, transition probabilities are green
* rewards are red
    * some are zero some are non-zero
* environment can transition to another state
* at \\( s_0 \\) you could take \\( a_0 \\) with prob 0.3 or other 0.7. Taking \\( a_0 \\) has prob 0.2 to go to \\( s_1 \\) or 0.8 to \\( s_1' \\)

## Reinforcement Learning

![](https://i.imgur.com/4pOOxOm.png)

* the model learns the policy!
    * \\( \pi_theta(a | s) \\)
* the probability is the state and action is 
* the goal is to find the parameters that maximizes expected reward
    * all rewards from transitions taken

## Value Functions

![](https://i.imgur.com/J9rVYCc.png)

* value of a state
    * sum for each action
    * policy (which is a probability) (think like after softmax before argmax)
    * times reward
    * plus sum of total future rewards
        * Value future reward
        * probability to enter that state
* it's recursive

## Belmman Updadte

![](https://i.imgur.com/jwTBMMz.png)

* Take the action that maximizes the term (reward + future scaled "expected" value from reward)

![](https://i.imgur.com/KYVHn4L.png)

* to get values, initialize each state value to zero, and update
    * state 1.0 get reward 1, etc.
* Then propogates the weights back 2.2 -> 2.0 because of 2.2 * .9 = 1.98 -> 2.0

![](https://i.imgur.com/PPtIYRq.png)

* But there can be cycles!
    * __not dynamic programming__
* if acyclic, then is O(SA) # states, # actions

![](https://i.imgur.com/VcuDz1x.png)

* infinite horozon vs finite horizon?

## Challenges of Reinforcement Learning

![](https://i.imgur.com/pLKiHis.png)

* the reward takes action which is discrete. We can't differentiate through this.
* Don't know the reward function
* `Temporal Credit Assignment Problem`
    * hundreds of actions to get to this step, they don't have any reward

## Policy Gradient Approaches

![](https://i.imgur.com/XCbcH5n.png)

* we can estimate the gradient

![](https://i.imgur.com/5mjNigf.png)

* we want \\( J(\theta) \\)
    * we can approximate it with samples \\(  \frac{1}{N} \sum_{i} ...\\)

![](https://i.imgur.com/H6DmBUp.png)

* \\( J(\theta) = \int \pi_{\theta} (\tau) r(\tau) d\tau \\\)
    * our policy network and the reward it would get
* to update our weights in the policy network
    * we can sample gradient log of our policy and the rewards it gets

![](https://i.imgur.com/JqZXAGu.png)

* our policy - based on all states and actions up to the current state and action
    * it is the probability of the initial state, and probability to get to our state
* expanded, they have no theta, so are zero.
* `solution` sample from 

![](https://i.imgur.com/8NZVd9n.png)

* run the policy
* get gradient which is possible on each action individually

![](https://i.imgur.com/cVo1C6S.png)

* minimizes gradient of crostt-entropy loss
    * as a neural network - forward state -> policy -> action
    * gradient correct action -> policy network
* ml problem of classification \\( s_t, a_t \\)

![](https://i.imgur.com/MbGYoCC.png)

* predicting a continuous value - find mean

## Reducing Variance

![](https://i.imgur.com/ICOSb7W.png)

* reward to go, Q function?

![](https://i.imgur.com/eXsugYo.png)

* subtract the baseline (average)
    * it's unbiased

## Off Policy Learning
* like from a human (not from the network's policy)

![](https://i.imgur.com/RprgdRc.png)

* from another distribution \\( q(x) \\)
* need a correction factor of L where expectation is 1
* just use them as a ratio

![](https://i.imgur.com/U4dNmCa.png)

* update using current policy but sampling from \\( \pi_{\theta'} \\)

## Challenges with Policy Gradients

![](https://i.imgur.com/F1pgAaQ.png)

* O(NT) - lots of steps
* Gradient estimate has hith variance

![](https://i.imgur.com/sHUPFjL.png)

* gradients steps expensive, minimize number of steps

![](https://i.imgur.com/h65y6vY.png)

* use new reward - maximize with penalty large change in \\( \pi_{\theta} \\) ?

## Trust Region Policy Optimization

![](https://i.imgur.com/IWJATL1.png)

* make sure the sample from reward in \\( \pi_{\theta'} \\) similar to \\( \pi_{\theta} \\)

![](https://i.imgur.com/AF8S8C2.png)

* uses a natural gradient like a newton step

## Proximal Policy Optimization

![](https://i.imgur.com/qXC1aK6.png)

* instead of KL divergence
* resists changes?
