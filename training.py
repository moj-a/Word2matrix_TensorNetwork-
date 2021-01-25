

import numpy as np
import random
import pandas as pd
from scipy import spatial
import matplotlib.pyplot as plt 


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# my own files:
from grammar import C_F_G
from voc import voc, tokenize_sentence
from model import binaryClassification
from utility import word2matr, get_input_layer, binary_acc

# ---------------Generation a list of 150 sentense about food

food_list=[]
food = C_F_G()
food.product('S', 'NP VP')
food.product('NP', 'N | ADJ N')
food.product('N', 'cook| chef')
food.product('ADJ', 'skillful | trained')
food.product('VP', 'VB NP2 | ADV VB NP2')
food.product('NP2', 'N2 | ADJ2 N2')
food.product('ADJ2', 'delicious | tasty')
food.product('N2', 'food | meal')
food.product('VB', 'prepares | serves')
food.product('ADV', 'carefully | attentively')

for i in range(250):
    food_list.append(food.rand_gen('S'))

# ---------------Generation a list of 150 sentense about IT

IT_list=[]
IT = C_F_G()
IT.product('S', 'NP VP')
IT.product('NP', 'N | ADJ N')
IT.product('N', 'programer| physicist')
IT.product('ADJ', 'skillful | smart')
IT.product('VP', 'VB NP2 | ADV VB NP2')
IT.product('NP2', 'N2 | ADJ2 N2')
IT.product('ADJ2', 'spotless | perfect')
IT.product('N2', 'code | program')
IT.product('VB', 'prepares | writes')
IT.product('ADV', 'carefully | eagerly')


for i in range(250):
    IT_list.append(IT.rand_gen('S'))

# -----------------Creating a list of 150 pairs of sentences and their associated labels
dataset=[]
i=0
while i < 33:
    # Selecting the two sentences from the two different lists (Food, IT 0)
    dataset.append([food_list[i], IT_list[i], 0])
    i += 1
while i < 75:
    # Selecting the two sentences from the two different lists (IT, Food, 0)
    dataset.append([IT_list[i], food_list[i], 0])
    i += 1
while i < 165:
    # Selecting the both sentences from the IT list (IT, IT, 0)
    dataset.append([IT_list[i], IT_list[i+1], 1])
    i += 2
while i < 224:
    # Selecting the both sentences from the food list (Food, Food, 0)
    dataset.append([food_list[i], food_list[i+1], 1])
    i += 2
# Shuffling the list
random.shuffle(dataset)

# Converting to numpy array
arr_dataset = np.array(dataset)

# Converting to csv file and save:
DF = pd.DataFrame(arr_dataset) 
DF.to_csv(r'/Users/moj/Desktop/dataset_grammar.csv')

# --------------Using voc class to find vocabulary list

VOC=voc()
vocabulary=VOC.voc_list(tokenize_sentence(arr_dataset[:,0])+tokenize_sentence(arr_dataset[:,1]))
word2idx=VOC.word2idx(tokenize_sentence(arr_dataset[:,0])+tokenize_sentence(arr_dataset[:,1]))

# Convert the vocabulary list to csv file and save:
DF2 = pd.DataFrame(vocabulary) 
DF2.to_csv(r'/Users/moj/Desktop/vocabulary.csv')

#-------------- Generating pairs (center word & context word)

window_size=2

idx_pairs= VOC.idx_pairs(tokenize_sentence(arr_dataset[:,0])+tokenize_sentence(arr_dataset[:,1]),window_size)


# Converting to numpy array
idx_pairs = np.array(idx_pairs)

#-----------------Word2matrix training-------------------------


# Embedding dimension 
embedding_dims = 5
num_epochs = 100
learning_rate = 0.001
vocabulary_size=len(vocabulary)


#  Two weight matrices
W1 = Variable(torch.randn(embedding_dims, embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims*embedding_dims).float(), requires_grad=True)


for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data, vocabulary_size)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())



        z1 = torch.matmul(W1, x)   

        # I flattened the embedding layer and convert it to a vector [5, 5]---> [5*5]  
        z2 = torch.matmul(W2, z1.view(-1))
    
        log_softmax = F.log_softmax(z2, dim=0)

        # Computing negative-log-likelihood on logsoftmax
        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.data.item()
        loss.backward()

        #  I wrote this by hand instead of creating optimizer object (SDG)
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        # Zero gradients to make next pass clear
        W1.grad.data.zero_()
        W2.grad.data.zero_()



# ----------------constructing the main dataset--------------


fe=word2matr(embedding_dims, word2idx, W1)

# Finding the similarity between two sentences 
X=[]

for i in range(len(arr_dataset[:,0])):
  X.append(1 - spatial.distance.cosine(fe.compo(arr_dataset[i,0]), fe.compo(arr_dataset[i,1])))
X=np.array(X)

# Labels
Y = np.array(arr_dataset[:,2]).astype(int)

main_data=np.array(list(zip(X,Y)))
# Convert the vocabulary list to csv file and save:
DF3 = pd.DataFrame(main_data) 
DF3.to_csv(r'/Users/moj/Desktop/main_data.csv')

# Spliting the dataset to two parts
X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.33, random_state=69)

# Standardizing features by removing the mean and scaling to unit variance

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 1))
X_dev = scaler.fit_transform(X_dev.reshape(-1, 1))

# --------------------------Hyper-parameters
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Managing data for dataloader

class Data(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_data = Data(torch.FloatTensor(X_train), 
                       torch.FloatTensor(y_train))

dev_data = Data(torch.FloatTensor(X_dev), 
                       torch.FloatTensor(y_dev))

# -------------------initialising dataloaders
# BATCH_SIZE = 1
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dataset=dev_data, batch_size=BATCH_SIZE, shuffle=True)

# Check device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# load data
model = binaryClassification()
model.to(device)
print(model)

# optimization 
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


#---------------Training--------------

model.train()


list_train_loss=[]
list_dev_loss=[]
list_train_acc=[]
list_dev_acc=[]
itera=[]
i=0

for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        

        # forward + backward + optimize
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    # Adding train_loss and train_accuracy for each epoch to the list     
    list_train_loss.append(epoch_loss/len(train_loader))
    list_train_acc.append(epoch_acc/len(train_loader))

    itera.append(i)
    i+=1


    dev_epoch_loss = 0
    dev_epoch_acc = 0

    with torch.no_grad():
      for X_batch, y_batch in dev_loader:
          X_batch, y_batch = X_batch.to(device), y_batch.to(device)
          
          
          y_pred = model(X_batch)
          
          dev_loss = criterion(y_pred, y_batch.unsqueeze(1))
          dev_acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        
          
          
          dev_epoch_loss += dev_loss.item()
          dev_epoch_acc += dev_acc.item()

    # Adding dev_loss and dev_accuracy for each epoch to the list
    list_dev_loss.append(dev_epoch_loss/len(dev_loader))
    list_dev_acc.append(dev_epoch_acc/len(dev_loader))



#-------------- Optimisation Plot
plt.plot(itera, list_train_loss, label="train")
plt.plot(itera, list_dev_loss, label="dev")
plt.title("Optimisation Plot (lr=0.001)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.show()

#--------------- Accuracy Plot

plt.plot(itera, list_train_acc, label="train")
plt.plot(itera, list_dev_acc, label="dev")
plt.title("Accuracy Plot")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend(loc='best')
plt.show()