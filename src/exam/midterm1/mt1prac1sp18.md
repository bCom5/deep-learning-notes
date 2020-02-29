# Midterm 1 Practice 1 Spring 2018
![](https://i.imgur.com/v1k6bSW.png)

1. a. Objects can be seen from different perspectives. Lighting can be different. Objects can be blocked; object partially visible.
    * __Review:__ `Lecture 1.`

    b. When doing transfer learning, the network has already extracted high level like edges, mid level like shapes and lines, and low level like faces. These generalize and don't have to be updated too much in being fine tuned. __The already trained low-dimensional features capture the essence of the data, can have faster fine-tuning.__
    * __Review:__ `Lecture 1, Transfer Learning.`

    c. Expected Risk is __expectation__ over all datasets of model loss (prediction – actual result). Empirical risk is difference over a fixed dataset sample. Machine Learning minimizes empirical risk.
    * __Review:__ `Expected Risk, Empirical Risk, Loss vs Risk.`

![](https://i.imgur.com/Z5g6JuI.png)

1. d. L2 is designed for loss of continuous value functions. Logistic regression is for binary classification (0 or 1). We use binary cross-entropy loss. (It encourages outputs to be 0 or 1?) The loss becomes a probability?
    * __Review:__ `L2 Loss vs Cross-Entropy Loss`
    * Why does it work for binary/discrete classification vs L2 for continuous? What is the relationship of it to probability?

    e. Newton's second-order method converge to local minima and saddle points. Where second derivative gradients (is this correct?) are zero
    * __Review:__ `Netwon Second-Order Methods`

    f. Using a max-margin classifier is the most robust (decreases overfitting) classifier to unseen data. It maximizes distance of the nearest points to the margin/decision boundary. Diagram shows decision boundary with max distance to nearest points
    * __Review:__ `SVM, Max-Margin, Hinge Loss`

![](https://i.imgur.com/ray5suc.png)

1. g. for each data point \\( (x_i, y_i) \\), we have \\( f_{y_i}(x_i) - f_j(x_i) \\) (the difference) for all classes \\( j \neq y \\). The original SVM loss is \\( max(0, 1 - yw^Tx) \\). Max margins for all classes not \\( j \\) is \\( max(0, 1 - f_{y_i}(x_i) + f_j(x_i) \\). Loss is sum/averaged. OvA classifer? What the heck is this?
    * __Review:__ `SVM`


![](https://i.imgur.com/yCRut4F.png)

1. h. Multiclass logistic regression can learn the multiclass naive Bayes
    * __Review:__ `Multiclass Logistic Regression, Multiclass Naive Bayes, their relationship`
    
    i. Gradient decreases proportionally to accumulating value at \\( \approx \frac{1}{\sqrt{t}}\\). This is because the denominator of the update contains sum of squares of past gradients.


![](https://i.imgur.com/0rGE4yp.png)

1. j. 
    * __Ans:__ Local optima were usually saddle points. SGD performs well because it does not just follow the gradient, so it is likely to fall off a saddle point.
    * __Corrections:__ 
    * __Review:__ `loss landscape and convexity, SGD` 

    k.
    * __Ans:__ Depends on the stride and padding. if stride 1 padding 0,  \\( 194 \times 194 \times 1\\)
    * __Corrections:__ 
    * __Review:__ `Convolution reshape equation` 

    l.
    * __Ans:__ They extract features in images. These features can be used for more robust representations of images.
    * __Corrections:__ 
    * __Review:__ `Convolutions, Feature Detection` 

    m. 
    * __Ans:__ Expectation of dropout is `output * p`. We can use the expectation of dropout in inference then, but also just use the `output` by doing dropout at training time (with a mask where each index has probability `p` being 1 (retained) and dividing by `p`.
    * __Corrections:__ 
    * __Review:__ `Dropout`

    n. 
    * __Ans:__ Prediction Averaging reduces variance. Each model is robust to different relationships (learn different things). Parameter averaging canceling out the relationships each model learns. Snapshot parameter ensembling works because the relationships the model was learning are close to the same, while still reducing variance.
    * __Corrections:__ 
    * __Review:__ `Prediction Averaging, Parameter Averaging, Snapshot Ensembling`

    o. 
    * __Ans:__ x -> [] (array with h wraps into block) -> y
    * __Corrections:__ 
    * __Review:__ `RNN`

    p. 
    * __Ans:__ `y = A_yh * tanh(A_hx * x + A_hh * h)`
    * __Corrections:__ 
    * __Review:__ `RNN equation`

    q. 
    * __Ans:__ Remember and Forget?
    * __Corrections:__ 
    * __Review:__ `LSTM equation and interpretation`
