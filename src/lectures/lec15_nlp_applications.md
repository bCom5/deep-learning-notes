# NLP Applications

## Review

![](https://i.imgur.com/5mGuBFY.png)

* There is self-attention
* Masked attention, for decoder, can only see in the past

## Generation vs Understanding
* NLG generate sequence as words
* NLU build a representation to be used for a task
* Summarization can be seen as both
    * Abstract Summarization
    * Extractive Summarization (3 most important, not generating text in this case!)

![](https://i.imgur.com/Rn21lmM.png)

* Pretraining: Language Modeling and Masked Language Modeling

## BERT

![](https://i.imgur.com/GEp1YXT.png)

* __Bidirectional__

![](https://i.imgur.com/5h9Shz5.png)

* Can't do Language Modeling, cause we already have the word! The model would cheat
* Instead mask some words. Guess the word!
* __Next sentence prediction__
    * Give Two sentences, does sentence two make sense to follow sentence one?
    * Learns to represent beyond the sentence

![](https://i.imgur.com/BycdfAl.png)

* Mask word and next sentence prediction

### Masked vs. Regular Language Modeling

![](https://i.imgur.com/xuqlfIZ.png)

* bi-directionality is important 50% to 65%. Having masks allows you to do bi-directionality

### Contextual Word Embedding

![](https://i.imgur.com/IFYurkY.png)

* `Run` has many meanings!
* Unlike word2vec, we use these contexts by going through stacked layers of self-attention and transformation

## BERT Task Specialization

![](https://i.imgur.com/GmFkv06.png)

* Segment Embeddings?
    * Dif Sentences

![](https://i.imgur.com/qMllq9L.png)

* Sentence Pair Classification
* Single Sentence like sentiment classification
* You can fine tune, it actually seems to work!
    * The residual layers can redirect to the relevant layers

![](https://i.imgur.com/85yLA3T.png)

* BERT did really well on GLUE.

## Tasks

![](https://i.imgur.com/LhCeK3Q.png)

* entailment, contracts, neutral
    * Kids play in the garden, vs no one goes to the garden
    * Can be ambiguous - "Do kids like playing Soccer?" 
* MultiGenre NLI
    * From Flickr and other datasets!

![](https://i.imgur.com/X2E7WAs.png)

* SQuAD, not part of GLUE but common question answering
* Big bias because the dataset always had an answer

![](https://i.imgur.com/S9TaXaF.png)

* SQuAD 2.0 (adversarial questions)

### Transfer Learning

![](https://i.imgur.com/pJ2sBgD.png)

* Transformer -> Pretrain on General Text -> In-domain text -> Fine-tune classification

### Evaluating Language Models - Perplexity

![](https://i.imgur.com/jtMBbLd.png)

* Uncertainty - normalize by length

![](https://i.imgur.com/sqQiATF.png)

* GPT-2 beat models __without finetuning__, low perplexity

### GPT-2 Generation

![](https://i.imgur.com/Bq29qiH.png)

### Multiple Languages

![](https://i.imgur.com/MIh8YiK.png)

* FAIR XLM did it with 15 languages

![](https://i.imgur.com/9dgRDQh.png)

* Translation Language Modeling
    * Has some English, some French, when masked can learn to use the other lang

### Conversational Q&A

![](https://i.imgur.com/MoUX4AG.png)

* Incomplete information