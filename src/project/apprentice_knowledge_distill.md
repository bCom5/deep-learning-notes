# Apprentice: Knowledge Distillation with Low-Precision Networks

[Paper](https://arxiv.org/pdf/1711.05852.pdf)

## Abstract

Low-precision numerics and model compression using knowledge distillation are popular techniques to lower both the compute requirements and memory footprint of these deployed model

Ternary Precision and 4-bit precision

Knowledge Distillation is used to "transfer knowledge" from the complex network to the smaller network.

## Three schemes
1. low-precision network and full-precision network
2. continuous knowledge transfer, - low-precision network trains faster
3. init with the full precision weights and gradually decrease precision

Hinton does knowledge distillation by diving the logits by a temperature before softmax. A higher temperature makes incorrect classes boosted.

## Knowledge Distillation

![](https://i.imgur.com/0cezU7F.png)

* \\( \alpha=1, \beta=0.5, \gamma=0.5 \\)

![](https://i.imgur.com/3ysbRRY.png)

* The third term and the line from student to teacher is the \ student network attempts to mimic the knowledge of the teacher network.

* We can change the params?

They use ternary and 4 bit precision


![](https://i.imgur.com/Ti1X2cy.png)

* Oh it's error rate. Accuracy is 100% - that value

![](https://i.imgur.com/tSaBnFj.png)

* Improves accuracy!

![](https://i.imgur.com/hnoA0iI.png)

* 

![](https://i.imgur.com/hnoA0iI.png)

*  27.2% Top-1 error 

![](https://i.imgur.com/cWYhF88.png)

* 32A, 2W (ternary?) and 8A, 4W