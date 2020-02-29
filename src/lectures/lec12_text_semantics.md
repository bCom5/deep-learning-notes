# Text Semantics

[Webcast](https://youtu.be/1rzjaUp6NiQ)

![](https://i.imgur.com/7LIXXrJ.png)

* __Propositional__: logic, formal, mathematical
* __Vector__: represent as numbers, high dimensional space.

![](https://i.imgur.com/br0ryjb.png)

* Propositional uses class (usually noun) and predicates - operation (usually a verb). __Reasoning__
* Word and sentences represented as vectors. Commutative! That is a problem.

## Vector Embedding of Words
![](https://i.imgur.com/CGwXg4v.png)

* __Bag-of-Words__ - sentence

### Word Similarity
![](https://i.imgur.com/Vd6TRq8.png)

* __Similar word__ in __similar contexts__. Dog and canine can be replaced.

![](https://i.imgur.com/Nqc0KTP.png)

### Dimension Reduction
![](https://i.imgur.com/L54iu21.png)

* Could __one-hot encode__ every word in vocab. That's expensive.
* Alt: counts of word used in that context. "Dog barks".


## Dimensionality Reduction
![](https://i.imgur.com/sd6BKtO.png)

### Latent Semantic Analysis
![](https://i.imgur.com/YogMF8j.png)

* Encode documents `N`
* `M` is word count.

![](https://i.imgur.com/hXnjAO7.png)

Counts words

### Latent Semantic Analysis
![](https://i.imgur.com/QEJ4hZq.png)

* Low dimensional approximation using SVD. (Review SVD!)
* Embedding is V, factors encode document contexts

#### Singular Value Decomposition
![](https://i.imgur.com/8vySkwT.png)

* \\( U, V \\) __orthogonal, normal__. \\( S \\) rectangular diagonal, singular values, right padded with 0s to fit

#### Review How to Calculate SVD
![](https://i.imgur.com/VHinLAL.png)

* It can be computed from eigenvalues of \\( T ~ T ^T \\)
* Insides cancel out, there's an \\( S^2 \\) term!!!

![](https://i.imgur.com/FB3xuv1.png)

Both work!

![](https://i.imgur.com/2ynYgmd.png)

* Can have a smaller S, using only the largest singular values (S?) 
* Small K dimension
* High dimensional T to low dimensional 
* __best possible reconstruction__ of documents from their embedding

![](https://i.imgur.com/mWKPVXo.png)

* Documents are T, Encoding is Z

![](https://i.imgur.com/0IJVSaK.png)

* Have a network learn SVD
* V is vocab times latent dimensions
* Scalable version of LSA/SVD
* similar things will get mapped to similar places


## t-SNE Word Embeddings
![](https://i.imgur.com/tF1VGAF.png)

## Word2Vec
![](https://i.imgur.com/LmTf2z2.png)

* neighborhood of a word instead of whole document, __skip-gram__
* nonlinearity

![](https://i.imgur.com/GLAWU4T.png)

* Predict center words from context
* order does not matter

![](https://i.imgur.com/ZrKymNE.png)

* Predict context words from center word

![](https://i.imgur.com/9sTa45d.png)

* Problem of SVD is it favors minimizing large distances (min squared error). Want to preserve __close__ distance like t-SNE

![](https://i.imgur.com/E3M523Q.png)

* Canny says it's a mess

![](https://i.imgur.com/29fFUrk.png)

* Holds relations as vectors! Vector math!

![](https://i.imgur.com/0WFGMaA.png)

* This model uses contexts
* City pairs, opposites, comparatives (great, greater)


Criticisms:
* Cross-entropy emphasizes small word combinations
* Expensive to softmax

![](https://i.imgur.com/Er3Ogvq.png)

* Maybe wrong data

![](https://i.imgur.com/CwVSFNj.png)

* Now words times words! Better for contextual words!
* Window size (prob exam problem), just try it and check

## GloVe
![](https://i.imgur.com/OkSCRaK.png)

* \\( C_{ij} \\) num times word j in context of word i. Query the matrix
    * For favoring the counts
* \\( u_i \\) is embedding, \\( v_j \\) is context word embedding
    * Inner product for similarity!!!
* \\( f \\) properties make it small for close words, and not too big for unlike words in context

![](https://i.imgur.com/CJ0MNrE.png)

## Compositional Semantics

![](https://i.imgur.com/NsrUYj7.png)

* __Compositional Semantics__ capture meaning in the structure and ordering 

![](https://i.imgur.com/H4Iio7l.png)

### Skip-Through Vectors
![](https://i.imgur.com/3TYFBkQ.png)

* Predict the previous sentence and next sentences
    * Fed back in
* RNN, each state 

![](https://i.imgur.com/ozMrn5g.png)

* Doesn't require backprop?

![](https://i.imgur.com/VCd42TW.png)

* Human evaluation.

![](https://i.imgur.com/LS5i6O1.png)

* Why not just train/optimize for similarity
    * Minimize Manhattan distance

### Semantic Entailment Evaluation
![](https://i.imgur.com/pHYzAi9.png)

* Tasks are Entailment, Contradiction, Neutral