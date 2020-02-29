# Midterm 1 Spring 2019

![](https://i.imgur.com/0WqLprF.png)

1. a.
    * __Ans:__ AlexNet is deeper. AlexNet is split on GPUs.
    * __Corrections:__ AlexNet uses ReLU, LeNet uses Sigmoid
    * __Review:__ `AlexNet and LeNet`

    b. 
    * __Ans:__ Multi-task learning is you replace the end layers of a network including the classifier, and can connect it to other tasks like question and answering. This makes it more robust to learn representations in images.
    * __Corrections:__ Layers are shared in a network (shared representation). The network is applied to multiple tasks. Benefits are: the network has extra information to capture the essence of a task, and avoids overfitting by learning more robust features.
    * __Review:__ `multi-task learning`

    c.
    * __Ans:__ Generative Models try to generate the joint distribution \\( P(B,A) \\). Discriminative models try to model only the decision boundary.
    * __Corrections:__ generate is "sample" data from distribution. Can't say it models \\( P(X, Y) \\). Discriminative doesn't need assumptions aboutd data, can model more complex distributions.
    * __Review:__ `Generative vs Discriminative Models`

![](https://i.imgur.com/mZLgVpH.png)

1. d.
    * __Ans:__ Cross Entropy is used for discrete values
    * __Corrections:__ Cross entropy loss (which is log loss in the binary case)
    * __Review:__ `Squared Error Loss and Cross Entropy Loss. Review Loss vs. Risk`

    e.
    * __Ans:__ (rip)
    * __Corrections:__ Loss gradients push the boundary to move points outside the margin. Close points have very large gradients.
    * __Review:__ `Logistic Regression, its training algorithm`

![](https://i.imgur.com/ZoKIsmv.png)

1. f.
    * __Ans:__ Bias is how accurate the model is on learning the relationship between x and y. Variance is how accurate the model is to data it has not seen. Deep neural networks have high variance and low bias.
    * __Corrections:__ Complex models can fit data better (low bias) but are more sensitive to data (high variance) Regularization reduces variance at expense for bias. Deep neural networks are low-bias, high variance
    * __Review__: `bias variance tradeoff, regularization, deep networks are low bias high variance`

    g.
    * __Ans:__ When each variable is independent. Logistic regression is more accurate when this is not the case and there is more data.
    * __Corrections:__ Naive Bayes assumes feature values are conditionally independent based on class label. Logistic regression doesn't make that assumption and is more accurate when that is not the case.
    * __Review__: `Naive Bayes, Conditional Independence on class label, Logistic Regression.`

![](https://i.imgur.com/h77LjNU.png)
1. h.
    * __Review:__ Gradients vanishes at local minima, maxima and saddle points
    * __Corrections:__
    * __Review:__ `Vanishing Gradients`

    i.
    * __Review:__ \\( 128 \times 64 \times 64 \\)
    * __Corrections:__
    * __Review:__ `Convolution Equation`

    j.
    * __Review:__ Split dataset into `k+1` subsets taking one out for testing. On round `i`, hold `i` for validation and and the rest of the `k` for training. Repeat with different holdout sets.
    * __Corrections:__ Check on fold `i` (validation data)
    * __Review:__ `Cross Validation`
