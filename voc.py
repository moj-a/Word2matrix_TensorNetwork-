
import numpy as np




#--------- Function for tokenization-----------

def tokenize_sentence(sentence):
    tokens = [x.split() for x in sentence]
    return tokens

#--------- Vocabulary class-----------


class voc:
    """A simple vocabulary class"""

    def __init__(self):
      self.vocabulary = []
      self.pairs_list = []

    def voc_list(self, corpus):
      for sentence in corpus:
          for token in sentence:
              if token not in self.vocabulary:
                  self.vocabulary.append(token)
      return self.vocabulary

    def voc_len(self, corpus):
      return len(self.voc_list(corpus)) 

    def word2idx(self, corpus):
      word2idx = {w: idx for (idx, w) in enumerate(self.voc_list(corpus))}
      return word2idx

    def idx2word(self, corpus):
      idx2word = {idx: w for (idx, w) in enumerate(self.voc_list(corpus))}
      return idx2word

    def idx_pairs(self, corpus, window_size):
      
      for sentence in corpus:
          indices = [self.word2idx(corpus)[word] for word in sentence]
          # for each word, threated as center word
          for center_word_pos in range(len(indices)):
              # for each window position
              for w in range(-window_size, window_size + 1):
                  context_word_pos = center_word_pos + w
                  # make soure not jump out sentence
                  if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                      continue
                  context_word_idx = indices[context_word_pos]
                  self.pairs_list.append((indices[center_word_pos], context_word_idx))

      return self.pairs_list