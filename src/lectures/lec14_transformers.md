# Transformers and Pre-Training

[Visual Attention](https://towardsdatascience.com/visual-attention-model-in-deep-learning-708813c2912c)

[Visualizing NMT](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

[Visualizing Transformers](http://jalammar.github.io/illustrated-transformer/)

## Review

![](https://i.imgur.com/BVdVEoi.png)

* Up and Down Attention

![](https://i.imgur.com/gGG4J6N.png)

* Transformers (Convolution of NLP)
    * Self Attention
    * Encoding/Decoding

## Transformers

![](https://i.imgur.com/mYSymyv.png)

![](https://i.imgur.com/5jvchbe.png)

* __Multi-Headed Attention__
    `h` block of scaled dot-product attention. Tensor V, K, Q

![](https://i.imgur.com/aqMQVVh.png)

* What is Encoder/Decoder?

### Transformer Encoder

![](https://i.imgur.com/Bd6mOCq.png)

* Has residuals of based inputs
* What is the Positional Encoding

![](https://i.imgur.com/FmYWjJH.png)

![](https://i.imgur.com/0dTtfWA.png)

* Why does it mix?
* From words and has positional encoding

### Multi-Headed Attention

![](https://i.imgur.com/LgGrwEe.png)

* Attention strength (sum) from different words

![](https://i.imgur.com/G490xoJ.png)

* Self Attention (?)
* What are the colors of the lines and on the words (?)

### Transformer Decoding

![](https://i.imgur.com/J7O9jOG.png)

* Decoding is to generate text
* Attention vs Self-Attention (?)
* What does the direction of the arrow mean?

![](https://i.imgur.com/IVhfqGJ.png)

* Green is (causal/masked) self-attention

![](https://i.imgur.com/NAR824z.png)

* At training time can do in parallel

![](https://i.imgur.com/7byCbh7.png)

* N-Best Transformers, k copy of your output word
    * renard with 0.3 confidence
    * canard with 0.1 confidence

### Transformer Position Encoding

![](https://i.imgur.com/vfl5qkP.png)

* The encoding doesn't have any ordering
* Position is a sinusoid?

![](https://i.imgur.com/AguYmZQ.png)

* If your vector is even or odd
    * Learnable shifting relative displacement
    * The inner product measures relative displacement

![](https://i.imgur.com/l2zCU1J.png)

![](https://i.imgur.com/oWqORx5.png)

* linear combination that is strongest at position (idk)

## Tokenization Challenges

![](https://i.imgur.com/zItfCEk.png)

* new word like Starliner? can use `UNK` char

![](https://i.imgur.com/twmby69.png)

* Break word down `##liner` from `starliner`

![](https://i.imgur.com/OFg3afu.png)

* Small vocab
* No `UNK` tokens

![](https://i.imgur.com/Qz0dYAU.png)

* Summarization, quadratic terms

![](https://i.imgur.com/oauz9sr.png)

* He mixed up M and N. Don't need `N^2` term, small `M^2` term

## Bert (Bidirectional Encoder Representations from Transformers)

![](https://i.imgur.com/Lkc4KuK.png)

* Language model. Text is like a label, so it is like self labeling! Predict next word, previous word, etc.
* Bert is a encoder model, bi-directional
* GPT is a decoder model

![](https://i.imgur.com/oYIMVEN.png)

* No encoder, no cross attention (?)

![](https://i.imgur.com/aNnUldP.png)

* Minimize language modeling loss

![](https://i.imgur.com/0mgUX62.png)

* Apply GPT to:
    * classification (what kind of speech)
    * Entailment (a implies b, contraction, independent)
    * Similarity has both orders
    * Multiple Choice, run it multiple times!

![](https://i.imgur.com/qVblTh7.png)

* More data, larger models keep getting better on performance!

## Summary

![](https://i.imgur.com/gyL8pqy.png)
