# Visualizing Transformers

[Link](http://jalammar.github.io/illustrated-transformer/)

## High Level

![](https://i.imgur.com/EjK8hNc.png)

* In the transformer black box we have an __encoder__ and __decoder__

![](https://i.imgur.com/DN6ElwE.png)

* The encoders are stacked on top of each other.

### Encoders

![](https://i.imgur.com/NZ8mlKa.png)

* The encoder block is made of a __self-attention__ layer and __feed-forward network__
    * self-attention helps the encoder look at words in the input sentence as it encodes a specific word

![](https://i.imgur.com/7W5GDqK.png)

* In NLP, each word is embedded as a vector
* list of word vectors is a tensor

![](https://i.imgur.com/E6136r3.png)

![](https://i.imgur.com/8jThN0h.png)

### Self-Attention

* The feed-forward layer is actually one network that is reused by each \\( Z_i \\) encoded vector. That is like a convolution

![](https://i.imgur.com/Vqn2VSb.png)

* Say you have the sentence "The animal didn't cross the street because it was too tired." What does __it__ refer to? In this case, "The animal." A self-attention layer lets the network learn this representation.

![](https://i.imgur.com/unpaPfS.png)

* We have the embedded words (green \\( X_i \\))
* For each word, we want to get out a __queries__ vector, a __keys__ vector, and a __values__ vector. We have learnable weights \\( W^Q \\), \\( W^K \\), and \\( W^V \\).
* The matrix multiplication of the green embedded word and the queries matrix, keys matrix, and values matrix create the queries vector, keys vector, and values vector for each word.

![](https://i.imgur.com/jsas4BF.png)

* For each word, we calculate a score using the queries vector and keys vector.

![](https://i.imgur.com/LpzehcO.png)

* We divide by 8 (sqrt of dim of keys vector for stable gradients) and take the softmax over all words (in a batch? in the corpus?)
    * (self-attention seems to be over a batch)

![](https://i.imgur.com/o2gTn7S.png)

* We multiply the softmax by the value and sum up all the weighted value vector \\( V_i \\)
    * (Does this mean all \\( Z_i \\) are equal though?)

### Self-Attention as Matrices

![](https://i.imgur.com/CmSRI5r.png)

* Batch of two words, embedded in a vector of size 2.
* (Each weight matrix is the same size though?)

![](https://i.imgur.com/eVmtrMH.png)

* Q, K, and V are calculated from X (They are NOT the weight matrices!)
* Their operation can be condensed into one formula

### Multi-head Attention

![](https://i.imgur.com/63q1zFr.png)

* Instead of just using one Q, K, and V weight we can use multiple!
* This is like multiple filters in a convolution.

![](https://i.imgur.com/vaVBv2C.png)

* To combine the multiple heads into a single head for the feed-forward network, we can concat all the heads \\( Z_i \\) and multiply it by a weight matrix \\( W^O \\)

![](https://i.imgur.com/vyxHEn7.png)

* Each step shown together
    * The input to self-attention does not have to be a batch of words (green matrix \\( X \\)) but it can be the output of an encoder below (blue matrix \\( R \\)). Encoders are stacked.

![](https://i.imgur.com/7RDKhZK.png)

* At the top is a bunch of colors. They represent heads. The blue head for the word "it_" has attention at "street", "'_", and "because". The green head has attention at "tire", "d_", and "was_".

### Positional Encoding

![](https://i.imgur.com/6nL9QBu.png)

* To encode position, we add positional encoding with our embedded word vectors

![](https://i.imgur.com/KJy5mug.png)

* The encoding is done with sin and cosine. Not much detail is added (@TODO go back to this in the lecture)

### Residuals in the Encoder

![](https://i.imgur.com/ObpOAHt.png)

* Like ResNet, Residuals are useful in Transformers.

![](https://i.imgur.com/856SQJe.png)

* The pre-attention vectors and post-attention vectors are added together and normalized like batch norm

## Decoder

![](https://i.imgur.com/ZttEnYU.png)

* The output vectors of the encoder are transformed into Key and Value matrices (not weights?) in the Decoder sequence
* The Decoder has to create it's own Queries matrix
* They are used by each Decoder (why?)

![](https://i.imgur.com/4O6WuNr.png)

* The previous input is fed into the decoder. The positional encoding is added again to this output

![](https://i.imgur.com/51kvlQj.png)

* The output embedded vector is matrix mutliplied to the vocab_size. It goes through a softmax, where the large index is the chosen word.

## Training

![](https://i.imgur.com/vlzyFKr.png)

* The data is matrix of batch size times one hot encoding and the true value is the same.
* In training we can keep the top k best predictions at each word. And feed best next words into the decoder. This is __beam search__
