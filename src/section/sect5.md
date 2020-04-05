# Section 5: Attention Mechanisms and Transformers

## Attention
* What is Attention?
    * it helps the network focus (weigh) parts of data more
    * in an RNN, this can focus not only on the past hiddel states

![](https://i.imgur.com/3WN6ja2.png)

* What is Luong Attention?
    * it computes alignment weights \\( a_t \\) with alignment socres into a context vector \\( c_t \\)

![](https://i.imgur.com/FdoBhEI.png)

* What is the output?
    * \\( \tilde{h_t}=\tanh(W_f \cdot [h_t ; c_t]) \\)
    * the \\( [h_t ; c_t]) \\) is concat

##  Scaled Dot-Product/Key-Query-Value Attention in Transformer Networks
* What Attention is in Transformers?
    * Scaled Dot-Product/Key-Query-Value Attention
    * It has K, Q, V

![](https://i.imgur.com/Fw10Ej9.png)


* What is the product of Query Q and Key K?
    * the  dot product is a score
* What is the interpretation of Q?
    * it is a querying term to find it's relation to K and V pair
    * we get a score for each K for each Q.
* Where do you get attention weights?
    * After softmaxing them all!
* What do you do with the values?
    * Multiply with the weights!

![](https://i.imgur.com/rpdlpIP.png)

* What is the correspondence to Luong Attention?
    * Q query <-> \\( h_t \\)
    * K key and Q query <-> \\( h_s \\)
    * V Value <-> \\( p_t \\)

* What is self-attention vs cross-attention?
    * Self-attention is just attention to this block
    * cross-attention is query is transformed output word embedding
    * key, value is trnasformed input word embedding

![](https://i.imgur.com/TLSIsP1.png)

![](https://i.imgur.com/Tcu7NnR.png)