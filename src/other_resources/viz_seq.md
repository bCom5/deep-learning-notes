# Visualizing Seq2Seq with Attention
[Link](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

## Review of Visual Attention

![](https://i.imgur.com/GpIFcPF.png)

* Have Image (top) and location `l_t-1` fed into `f_g` glimpse network.
* This is fed into a recurrent network `f_h`
* The output hidden state is fed into two networks
    * `f_a` activation network which does the prediction
    * `f_l` location network which predicts the next location `l_t`. In Soft Attention this is done with a saliency map.

![](https://i.imgur.com/3lZj6UK.png)

* The model has an __encoder__ and __decoder__
* __Context__ is transfered from the encoder to the decoder
    * Context is a vector of floats. It is a hidden state in a RNN

![](https://i.imgur.com/IVCpWGs.png)

* The words are embedded as vectors using a __word embedding__ algorithm.

### At Attention!

* __Attention__ is the idea some words are more important than others when getting the translation
    * Café to coffee

![](https://i.imgur.com/VviBRGt.png)

* Instead of one hidden state, pass all the hidden states!

![](https://i.imgur.com/R0V8kgM.png)

* Use the decoder hidden state to score encoder hidden states. This is  __attention__.

![](https://i.imgur.com/a012seQ.png)

* `h_4` : hidden state and `c_4` : context state are concated and the output word comes out
* Where does the scoring happen though?

## From Lecture

![](https://i.imgur.com/ZETrFIt.png)

* In lecture, it is weighed alignment scores
* alignment score: \\( \text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \mathbf{v}_a^\top \tanh(\mathbf{W}_a[\boldsymbol{s}_t; \boldsymbol{h}_i]) \\)
    * \\( \mathbf{v}_a \\) and \\( \mathbf{W}_a \\) are weights that can be learned

![](https://i.imgur.com/RFYS1Xt.png)

* Combine current recurrent state and all input states into attention weight (soft-attention) of input states. The attention weight adds up to 1, and is done with a softmax. 
