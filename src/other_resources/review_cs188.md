# Review CS 188: AI

[Link](https://docs.google.com/presentation/d/1KVQq4llaJA7beVuIhE-8DvFKbq1w5Hdtjuzv4nYkJZc/edit?usp=sharing)

## Markov Decision Processes
* Starts on Slide 38

![](https://i.imgur.com/Kj265qE.png)

* states, choose an action (slow or fast), then it __non-deterministic__ goes to another state. reward gained when entering the state
    * look at overheated, -10 reward!


![](https://i.imgur.com/k3mmXTH.png)

* If you know the reward and transition functions
    * You can solve the MDP with Value/Policy Iteration
* Else you solve it with Reinforcement Learning
    * Model Based RL or Model Free RL

## Bellman Equation

![](https://i.imgur.com/0rCUf7s.png)

* \\(V^{*}(s) \\) is the __value__ of a state if acting optimally
* \\(Q^{*}(s, a) \\) Q is the value of taking action a in state s, and acting optimally after
* \\( V^\*(s) = \max_a Q^\*(s, a)\\)

![](https://i.imgur.com/CWULp54.png)

* At the end of the V and Q equation is a V and Q. They are equivalent

## Value Iteratio

![](https://i.imgur.com/Wao7fpm.png)

* N iterations, recursively solve down the graph. converges to correct answer

## Policies

![](https://i.imgur.com/QmLvPp1.png)

* \\(V^\pi\\) value of state using policy \\( \pi \\). It determines the action
* now do a single action instead of all actions
* goal learn best policy!

## Optimal Policy

![](https://i.imgur.com/dAdpszv.png)

* the optimal policy chooses the action that maximizes future reward?
* the optimal policy value should equal the optimal value?

## Policy Iteration

![](https://i.imgur.com/gp22GHc.png)

* inner loop learns the value of policy pi.
* outerloop update for best policy pi

![](https://i.imgur.com/EPwLMvg.png)

## Reinforcement Learning
* when you don't have reward and transition functions 
### Model Based RL
* try to estimate
    * transition func: \\( T(s, a, s') \\)
    * reward func: \\( R(s, a, s') \\)
* now do policy iteration or value iteration!

### Model Free RL
* basically yolo and learn
* "take an action and __see what happens__"
* learn what actions are good even if you don't know __why__ they are good
    * (don't know the transitions and rewards)

### Temporal Difference (TD) Learning
* passive (fixed policy) RL

![](https://i.imgur.com/5FRBdk5.png)

* just take a sample of state, policy, next state, and reward
    * v is reward and discount of future value
    * update our policy value for that state with an exponential moving average
* good for learning V-values but not policies
* does not track __all__ v-values
* doesn't know about unvisited states

### Q-learning
* active (online?)

![](https://i.imgur.com/uS6KjNe.png)

* update Q

### Exploration vs Exploitation
* epsilon-greedy, random policy with small epsilon, our learned policy wtih 1-epsilon

### Approximate Q-Learning

![](https://i.imgur.com/w4eNIMu.png)

* __feature based Q-learning__

![](https://i.imgur.com/CoG6arI.png)

* don't learn Q-Values directly learn the weights

![](https://i.imgur.com/x28q7v3.png)

* find the difference and update the q function?

![](https://i.imgur.com/Qc4a7IE.png)

* instead use the difference to learn the weights

## Questions
* Is Online On-policy and off-policy?
