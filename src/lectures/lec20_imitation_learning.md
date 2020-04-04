# Imitation Learning

* Fairness
    * Unawareness
    * Domographic Parity
    * Separation
    * Sufficiency

* Causality
    * it's hard to figure out hte dependency
    * disease such as for pregnancy does have bias to gender

## Imitation Learning
* Robot
* Imitate Human actions
    * __behavior cloning__ also known as __sensorimotor learning__
    * end-to-end vision
* Markov Decision Process
    * \\( s_t \\) - state
    * \\( a_t \\) - action
    * \\( r_t \\) - reward
        * deterministic
    * \\( \pi_\theta (a_t | s_t) \\) - policy: probability
* For Partially Observable
    * \\( \pi_\theta (a_t | o_t) \\) - policy: probability of action \\( a_t \\) given obseration \\( o_t \\)
    * \\( o_t \\)


![](https://i.imgur.com/jDI1ZiI.png)

* errors accumulate

![](https://i.imgur.com/x6hT1n3.png)

![](https://i.imgur.com/lhpVC4R.png)

* left, center, and right camera

![](https://i.imgur.com/Fs8e7Yv.png)

* the right camera gives the signal turn left

![](https://i.imgur.com/NM18VPb.png)

* behavior policy what the agent uses
* target policy - ?

![](https://i.imgur.com/HOapigK.png)

* states from policy
* but labels from human

![](https://i.imgur.com/wpRSTbW.png)

* but have to get policies from a human

## GAIL: Generative Adversarial Imitation Learning

![](https://i.imgur.com/dfseUk0.png)

* __Inverse Reinforcement Learning (IRL)__
    * estimate cost function (loss function or -reward function)

![](https://i.imgur.com/61l56O0.png)

* under constrained - use regularization heuristics
    * minimize entropy regularization
    * learn the cost function
    * that we know the cost function the RL policy learning problem uses the new cost function

![](https://i.imgur.com/uf7Ww4C.png)

* low level behavioral policy

![](https://i.imgur.com/uf7Ww4C.png)

* __on-policy__
* GAIL, more powerful find the cost function then use RL
