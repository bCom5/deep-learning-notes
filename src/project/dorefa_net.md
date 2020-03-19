# DoReFa-Net

## Abstract
* low bitwidth weights and activation
    * 1 bit weights, 2 bit activations
* low bitwidth gradients

![](https://i.imgur.com/isbpqr4.png)

* bitcount when both weight and input activations are binarized (XNOR-net 2016).
* 1 bit convolution kernels?
* 1 bit weights, 1 bit activations and 2 bit gradients do well

![](https://i.imgur.com/isbpqr4.png)

* x and y are arrays of bits, the floating point can be calculated by those sums.

![](https://i.imgur.com/UKMw0Vt.png)

* dot product

![](https://i.imgur.com/aagv0oS.png)

* Straight through estimator through sampling and quantization

![](https://i.imgur.com/77wxwFU.png)

* k bit representation with tanh to constraint to [-1, 1]

![](https://i.imgur.com/OuygHvF.png)

* .975 with all accurcay
* .934 with 1W, 1A, 1G
* .971 with 1W, 1A, 32G
* W,A =(1,2) and G >= 4 is best

Quantizing the first and last layers leads to significant degradation 

FPGAs with B-bit arithmetic is good at low bitwidth convolutions