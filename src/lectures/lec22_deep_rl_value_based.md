# Deep RL: Value-Based Method

![](https://i.imgur.com/P8oAd1Y.png)

* theta is weights, goal is to optimize total sum reward

![](https://i.imgur.com/FKp4duA.png)

* get policy gradients. Take multiple samples and calculate the gradient
* but it is expensive has to take many steps
* TRPO and PPO (look at amount of change)

## Value Function

![](https://i.imgur.com/g4R3YEZ.png)

* value is the discounted reward-to-go
* value is the probability of taking that action, actual reward, and discount of future total reward

![](https://i.imgur.com/S0dE5JJ.png)

* \\( Q^{\pi}(s, a) \\) expectation on both state and action
* the value function is the expectation oe all future discounted rewards
    * expectation of a sampled from Q, for all 
* advantage function, reduces magnitude, lower variance

## TRPO and PPO use Advantage

![](https://i.imgur.com/bhqJUpo.png)

* \\( A^{\theta} \\) at the end

## Actor-Critic Introduction

![](https://i.imgur.com/ivQDaNY.png)

* actor: policy
* critic: value estimator

![](https://i.imgur.com/dPMqdEi.png)

* compute value func and advantage func (Q func implicitly)
* sample by running on our robot
* online:
    * sample action from policy
    * update V with target?

![](https://i.imgur.com/EmcInDw.png)

* you can just take the best action

![](https://i.imgur.com/2lOOQ64.png)

* finding best in action function, you can actually just use the Q-func

## Fitted Q-iteration

![](https://i.imgur.com/3qwZu42.png)

* what is \\(y_i\\)  (temp)
* find \\( \rho \\)

## Off-Policy vs On-Policy State Visitation

![](https://i.imgur.com/JYPagBL.png)

* can easily diverge

![](https://i.imgur.com/afiMaba.png)

* iteratively fit, using final game state (final reward)

## Online Q-learning

![](https://i.imgur.com/kwCf1PT.png)

![](https://i.imgur.com/FrBqFlE.png)

* it's deterministic prob are 0 or 1.
* use epsilon greedy - now probabilitic and learns

## Correlation Problem

![](https://i.imgur.com/jvC02fP.png)

* states are correlated 

## Another Solution: Replay Buffers

![](https://i.imgur.com/2KOMZ9k.png)

* save tuples from dataset. They are less likely correlated

![](https://i.imgur.com/4We9ue8.png)

## Q-Learning with Target Networks

![](https://i.imgur.com/dOC1zZY.png)

## Classic DQN (In Homework)

![](https://i.imgur.com/MkxJgy7.png)

* theta for network weights, rho is the input, 84x84 4 frames

![](https://i.imgur.com/FD7So4H.png)

* __frame history state__

![](https://i.imgur.com/FD7So4H.png)

* convolution layer

![](https://i.imgur.com/MzS5Jqj.png)

* usually an overestimate

![](https://i.imgur.com/qw2YSst.png)

## Double Q-Learning

![](https://i.imgur.com/MaYvz9R.png)

* instead of a (action) use the best action from another policy! It's unbiased!

![](https://i.imgur.com/wmjR4uh.png)

* use the __current network__
* and __target network__ which is just delayed

![](https://i.imgur.com/Exs67av.png)

![](https://i.imgur.com/mKQNR2s.png)

* ? 

![](https://i.imgur.com/Gdxnq3A.png)

*  rainbow

## Professor John Canny on a Beach

![](https://i.imgur.com/MYLYwR7.png)
