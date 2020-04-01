# Improved Knowledge Distillation via Teacher Assistant

* [Link](https://arxiv.org/pdf/1902.03393.pdf)

* However, we argue that knowledge distillation is not always effective, especially when the gap (in size) between
teacher and student is large

* Hinton, et. al (2015) do knowledge distillation from the logit with a tempature function.
    * The so-called dark knowledge transferred in the process helps the student learn the finer structure of teacher network. Hinton, Vinyals, and Dean (2015) argues that the success of knowledge distillation is attributed to the logit distribution of the incorrect outputs, which provides information on the similarity between output categories.

## Knowledge Distillation

![](https://i.imgur.com/I3TbjGP.png)

* \\( a_s \\) is logit (before softmax) of student, \\( y_r \\) is true label

![](https://i.imgur.com/prFCVW1.png)

* The student tries to match student and teacher logits.
    * \\( y_s = \text{softmax}(\frac{a_s}{\tau}) \\)
    * \\( y_t = \text{softmax}(\frac{a_t}{\tau}) \\)
* KL divergence tries to match the two distributions

![](https://i.imgur.com/5mb4wyg.png)

* The final loss function

![](https://i.imgur.com/SqMzJ7s.png)

* Teacher accuracy increases as teacher size increases yet student size decreases
* Solution: Use TA networks in between to smooth the transition!

## What is the best TA size?

![](https://i.imgur.com/C6cWTkf.png)

* mean the network size is good

![](https://i.imgur.com/CxjetZa.png)

* Highest accuracy is most distillation steps