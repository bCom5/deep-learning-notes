# Visualizing BERT, ELMo, and GPT

[Link](http://jalammar.github.io/illustrated-bert/)

![](https://i.imgur.com/vaB3saH.png)

* Step 1 is to semi-supervise on large amounts of task. Usually predicting masked words.
* Step 2 is to fine grain train on a dataset

![](https://i.imgur.com/ICfXukJ.png)

* BERT then with a fine grain spam classifier

![](https://i.imgur.com/R5YLWxl.png)

* BERT is a encoder transformer stack

## Word Embeddings and ELMo

![](https://i.imgur.com/wNFFOZc.jpg)

* ELMo uses contextual (where in the sentence) to embedded.

![](https://i.imgur.com/Ynzilsc.png)

* ELMo predicts the next likely word

![](https://i.imgur.com/ZNU7ol2.png)

* ELMo also includes information of the next words in a sentence/batch. The forward and backward LSTM are concatenated together

## OpenAI Transformer
* It uses only the transformer decoder

![](https://i.imgur.com/7feXECY.png)

* Predict the next word

![](https://i.imgur.com/ddFdSrJ.png)

* Transfer Learn!

![](https://i.imgur.com/bUgDTmk.png)

* More Transfer Learning. Often with tasks, the structure of inputs is important

## BERT

![](https://i.imgur.com/MtcxjTT.png)

* 15% of words are masked. BERT tries to predict the masked word

![](https://i.imgur.com/NzChyLx.png)

* An additional task was to predict if a given sentence was followed by another given sentence

![](https://i.imgur.com/UN4nCpS.png)

* BERT transfer learning

![](https://i.imgur.com/4kqLRhK.png)

* BERT can do word embeddings

![](https://i.imgur.com/qB1f52C.png)

* F1 score decreases the deeper the network. Deeper should give better embeddings.