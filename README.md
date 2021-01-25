# Word2matrix_TensorNetwork-
A simple exercise of supervised learning using Word2Matrix embedding and Tensor Network for NLP:



At first, I created a dataset of short sentences falling into two categories (IT and food) for a the following simple contex_free grammar:


S → NP VP
NP → N
NP → ADJ N
VP → VB NP
VP → ADV VB NP


My dataset has the form of:

(sentence1 sentence2 label)


Where the label is 1 if the two sentences are of the same category (they both talk about food or both talk about IT) and 0 otherwise.

Then I created a word to matrix embedding and I used Tensor Network to find a vector representation of each sentence.

The rest of it is a simple supervised classification!


