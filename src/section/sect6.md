# Section 6: Transformers and Pretraining in NLP

## Transformers
* What are the parts of a transformer?
    * word embeddings
    * positional embeddings
    * self-attention
    * cross-attention
    * feed-forward layers

* What inputs does a transformer receive
    * two sequences x_source and x_target
* What types of transformers are there?
    * encoder attention (non-casual self-attention)
    * decoder attention (casual self-attention)
    * encoder-decoder attention (cross-attention)

![](https://i.imgur.com/mlNcLcg.png)


## Transformer Encoder
* Input + Positional Embedding
* we use positional encoding to differential sequence positioning
* What is the attention update function?

![](https://i.imgur.com/FQnyA88.png)

* What is multihead?
    * has another tensor dimension!
    * \\( B \times L_{source} \times D \\)
    * to \\( B \times H \times L_{source} \times D/H \\)
    * add negative infinity for padded positions?

## Feedforward layer
* What is a feed forward layer?
    * literally a linear transformation (linear layer)
* Where is most of the computation?
    * in the feed forward layer!

## Transformer Decoder
### Masked Decoder Self-Attention
* What is masked decoder self-attention?
    * We want each position in the sequence to pay attention to previous positions but not future positions.
* how do we do that?
    * if i <= j set the score to q_i^T * k_j / sqrt(D) else -inf

## Encoder-Decoder Attention
* Aka cross-attention
* What is the difference to other attentions?
    * one attention generates queries (from X_target)
    * one attention generates keys and values (from X_source)
        * K = X_source * W_K
        * V = X_source * W_V

## Pre-training in NLP
* what is the generic pre-training task?
    * given large __unlabeled__ text, predict next word or predict missing word
* What are the two main models?
    * BERT (Bidirection Masked-Language Modeling)
    * GPT-2 (left to right)
### 

![](https://i.imgur.com/Bs1c6Xo.png)

* 4 different "adapted" tasks
### GPT-2
* predicts next word
* can generate words!