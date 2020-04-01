# Fairness in Deep Learning

Deep Learning has bias

![](https://i.imgur.com/HeZAFL6.png)

* Universal Approximation, but
    * the data we train can have __sampling bias__
        * Australians are under-represented
    * __selection bias__
* But actually want bias to be equal to variance

## Unawareness in Fairness

![](https://i.imgur.com/eNJiAGY.png)

* \\( A \\) is protected attribute
* we want __unawareness__, \\( f(X, A)=f(X) \\)

## Demographyc Parity

![](https://i.imgur.com/vK5rwFn.png)

* \\( P(\hat{Y}=1|A=Australian) = P(\hat{Y}=1|A=American)\\)
* The prediction is independent?
* There is a notation

![](https://i.imgur.com/sZE9srs.png)

* somehow unfair

## All Other Things Being Equal

![](https://i.imgur.com/ALTQ9ZI.png)

* Think about hypothetical outputs
* `Causal Inference`

## Separation

![](https://i.imgur.com/3EggPu1.png)

* Separation is Conditionally Independent given \\( Y \\)

![](https://i.imgur.com/LyiAGIk.png)

* Constrain it more such that \\( Y=1 \\)
    * Only be fair for people worthy of loans

## Sufficiency

* Too complicated

## Causal Reasoning

![](https://i.imgur.com/Puzi4he.png)

* Graphical Models, for factorization of joint distribution

![](https://i.imgur.com/3nDF36r.png)

* as an algorithm
    * sampling then joint distribution

![](https://i.imgur.com/1tQytMq.png)

* `intervention`

![](https://i.imgur.com/1tQytMq.png)

* in a graphical model there is a lot of other indirect effects

## GAN like optimization

![](https://i.imgur.com/8oXEGK6.png)

* with a Jenson Shannon Divergence
    * optimize distribution similarity

![](https://i.imgur.com/74zJRxd.png)

* Y, and X
* bottom is for fairness, protected value
    * switch between real vs fake

![](https://i.imgur.com/0T9b1gH.png)

* lower bound

![](https://i.imgur.com/4uKD2aZ.png)

* Conditional Mutual Information

## Summary

![](https://i.imgur.com/x5JfLNm.png)

* 
