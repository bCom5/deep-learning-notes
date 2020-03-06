# Translation

* [Slides](https://briantliao.com/store/cs282-lectures-slides/lec13.pdf)

## Review

![](https://i.imgur.com/ytGyXuo.png)

* Inner Product (or is it Dot Product?) is closeness

![](https://i.imgur.com/E9DMaxX.png)

* Predict previous and next sentences, now ordering is included

![](https://i.imgur.com/r9NLuPf.png)

* Train for your task! This is based on closeness.

## Translation

![](https://i.imgur.com/myVDtam.png)

### Sequence to Sequence
* __Sequence-To-Sequence RNN__, input fed into left, output comes out of the right

![](https://i.imgur.com/3HwFFEZ.png)

* At each RNN node, the output is fed back in. Keep the n-best

### Bleu

![](https://i.imgur.com/9G6bB4e.png)

* Candidate is what our network gives us. References are from humans.
* See where it matches.

![](https://i.imgur.com/1lMHytF.png)

* Unigram matching - mapping

![](https://i.imgur.com/j2kCRM2.png)

* Averaged over references

![](https://i.imgur.com/IyG2pdL.png)

* __Unigram__ for __adequacy__
* __Ngram__ for __fluency__
    * Fluency is better (imo)

![](https://i.imgur.com/j4kbkLo.png)

* Bigram (2 word length)

![](https://i.imgur.com/P9iowOf.png)

* Geometric Avg ( \(( w_n log p_n \\) )
    * BP: penalty shorter than r

![](https://i.imgur.com/4xTSym1.png)

* Ensemble does well

![](https://i.imgur.com/4dQNLlc.png)

* Try going backwards!
    * coffee love I

![](https://i.imgur.com/EC4HORT.png)

* Really small BEAM search

![](https://i.imgur.com/tJE0USb.png)

* Problem! There's a bottleneck for information

![](https://i.imgur.com/qERSjqN.png)

* In soft-attention, Coffee is related to cafe
    * But bottlenecked!

## Soft Attention for Translation

![](https://i.imgur.com/ZETrFIt.png)

* __Context vector__ is the sum of all weighed h, which are hidden states
* Weights are __Mixture weights__, softmax over alignment scores
* __Alignment scores__ input words and output words?

![](https://i.imgur.com/YPalgse.png)

* Wow this is amazing
* Bi-directional RNN Encoder

![](https://i.imgur.com/Z1r7qjJ.png)

* Decoder is a RNN, sample word fed back in, get next word out
* You have a __Recurrent State__ in the __Decoder RNN__
* and a __Attention Vectors__ hidden state in the __Bidirection Encoder RNN__

![](https://i.imgur.com/RFYS1Xt.png)

![](https://i.imgur.com/nl7EEHM.png)

* English to French, correctly picks up reversal!

![](https://i.imgur.com/EF8asQs.png)

* __Neural Machine Translation!__
    * Complicated model, what the heck

### Stanford Manning Update Attention Neural Machine Translation

![](https://i.imgur.com/LU9UKYM.png)

* Stacked LSTM

![](https://i.imgur.com/GyKZ5bK.png)

* __Global Attention__, Attention is not Recurrent, align weights are now global

![](https://i.imgur.com/4AXBT7p.png)

* Alignment (?)

### Translation and Parsing

![](https://i.imgur.com/EqGiSWf.png)

* Can generate like nested/tree code!

![](https://i.imgur.com/I2NfCBU.png)

* Build the parse tree (Nouns, parts, etc.)
    * Trees are like LISP, can be nested in parenthesis

![](https://i.imgur.com/u5gQqQI.png)

* Sequence-To-Sequence Parse Tree

![](https://i.imgur.com/SBY9QQq.png)

* Training, on the three-bank not well
* Then train on the Berkeley Parser
* Add attention and retrain on human data
    * It's like Model Distillation?
    * Something about overfitting

## Attention-only Translation

![](https://i.imgur.com/wNITJAH.png)

* Time grows in proportion to sentence length
* Long-range hard
* Hierarchal deeper structure hard

### Transformer

![](https://i.imgur.com/YJD3LBN.png)

* Query-Key-Value

![](https://i.imgur.com/ki9Z54a.png)

* Try my query to other keys

![](https://i.imgur.com/ctviBRe.png)

* Score is local, Softmax x Value then Sum is global

![](https://i.imgur.com/079hOLX.png)

* Input x, compute inner products with their matrices
    * \\( W^V, W^K, W^Q \\)
    * Get \\( V_1, K_1, Q_1 \\)

![](https://i.imgur.com/3Xd2tFj.png)

![](https://i.imgur.com/SUfx7a8.png)

* Q, K, V are matrices now

![](https://i.imgur.com/qzELmTE.png)

* __Multi-Headed Attention__

![](https://i.imgur.com/14NfcZi.png)

* multiple heads
    * recognized more difficult is coupled
    * blue, green, red

![](https://i.imgur.com/sPO4eMU.png)

* __Transformer Encoder__
    * Encoded as a single matrix

![](https://i.imgur.com/jTD6Qo6.png)

#### Transformer Encoder/Decoder

![](https://i.imgur.com/IKZ7vfD.png)

* Value Key fed in

![](https://i.imgur.com/wZmiJ9e.png)
