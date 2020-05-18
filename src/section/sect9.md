# Section 9: Policy Gradient and Value-based Methods

## Policy Gradient Methods
* Why do we have Policy Gradient Methods?
    * We may not have access to expert data. We need to learn from the environment
* How can we do that?
    * optimize \\( \theta \\) in \\( \pi\_{\theta} \\)
    * Given a state predict action to take
* Why can't we differentiate with reward with respect to theta?
    * The reward is a function of action r(a), a is discrete
    * we don't know what the reward function is r(a)
    * __temporal credit assignment problem__, state depends on previous actions, but which previous action produced the reward?

## REINFORCE
* What is REINFORCE?
    * We can estimate the gradient sampling several trajectories

![](https://i.imgur.com/hik1gRz.png)

* The optimal theta is the theta that maximizes the expected sum of reward along a trajectory that has probability given by the policy?
* this is approximately the average of the sum of rewards along multiple trajectories

![](https://i.imgur.com/hS4KLBb.png)

![](https://i.imgur.com/LQJpeXB.png)

* rewriting it and expanding the expectation is a an integral of the probability output by the policy and the reward that policy gets
