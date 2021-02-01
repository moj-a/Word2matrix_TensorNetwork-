

import numpy as np
import torch



# ----------Encoding the center word in one-hot manner, Function for input layer------------------

def get_input_layer(word_idx,vocabulary_size):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

#------------------ word to matrix function ------------------------

class Word2Matr:
  """This class dives the matrix embedding of words, 
     and also construct MPS for sentences based on a very simple tensor network structure.
  """
    
  def __init__(self,embedding_dims, word2idx, W1):
      """
      param word2idx:  The dictionary of word and id for our vocab
      param embedding_dims: The embedding dimention of the word to matrix
      param W1: The first weight matrix of the word2matrix embedding
      
      """

      self.embedding_dims = embedding_dims
      self.word2idx= word2idx
      self.W1 = W1

      # Special token
      self.sp = np.ones(self.embedding_dims)


  def get_embedding(self, word):
      
      
      try:
          idx = self.word2idx[word]
      except KeyError:
          print("`word` not in corpus")
      idx = self.word2idx[word]
      # swap dimensions
      sW1=self.W1.permute(2,1,0)
      return sW1[idx]
        
  def compo(self, sentence):
      """
      Composition of the matrix reperesntation of the words into sentnece
      """
      words = sentence.split()

      # identity matrix
      idn=np.identity(self.embedding_dims)
      for word in words:
        # Multiplication of matrices
        feature_mtx = idn.dot(self.get_embedding(word).detach().numpy())

        # Multiplying with the special token
        feature_vec=self.sp.dot(feature_mtx)
      return feature_vec

# --------------Accuracy function--------------------


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


