#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib as plt 

#Note on loading torch_geometric: If using jupyter notebook I had repeated crashes if the torch geometric modules were not loaded first so if you have that problem make sure to load them first in your file 
from torch_geometric.loader import DataLoader #This imports the data loader which is used for dealing with the data involved in the different files, basically rather than using a generator this performs the work for us by only picking out some files to be loaded from a list.


from sklearn.model_selection import train_test_split#This is used to split the data into validation and training data sets 


from torch_geometric.data import Data#Data is a type of object structure that is used for dealing with graphs in pytorch 

from torch.utils.data import Dataset #Dataset contains a bunch of methods for dealing with a set of data 

from torch_geometric.nn.norm.batch_norm import BatchNorm#Batch norm is batch normalization that can be used to normalize some of the input graphs 

import os 
import numpy as np

from zipfile import ZipFile
import zipfile 

import torch

from torch.nn import Linear#This import a simple linear function 

import torch.nn.functional as F #Functional is just a way of loading a bunch of functions into the program such as dropout 

from torch_geometric.nn import GCNConv, GraphConv, GravNetConv, ChebConv # A series of different of different methods for graph convolution classification, they all have slightly different implmentations so I recommend checking the pytorch geometric documentation but they are all worth trying 

from torch_geometric.nn import global_mean_pool #Acts on the entire graph to pool the final output graph features 


filenames  = []#Save all the filenames somewhere 


train_dir = '/Dataset/'#Whichever directory all of the graph data is saved in 
k = 0

for subdir, dirs, files in os.walk(train_dir):#This will walk through the directory and pick out the file names 
    for file in files:
        filenames.append(file)
        #k = k+1#If you want to work with a subset of data you can use k to count to however many files you want to include 
        #if k==101:
        #     break
print(len(filenames))



TrainFiles, TestFiles = train_test_split(filenames,train_size=0.7,test_size=0.3)#This splits the data into the validation and training dataset 

print(len(TrainFiles))#Gives the number of files in each data set, useful to estimate time to train 

print(len(TestFiles))

TrainD = DataLoader(TrainFiles, batch_size=64,shuffle=True)#This loads the filenames and shuffles them with a batch size of 64 for the training dataset 

TestD = DataLoader(TestFiles, batch_size=64,shuffle=False)#Loads the validation filenames but does not shuffle them 







def train():#This function will be responsible for training the network 
    model.train()#This tells the model to enter training mode 

    for data in TrainD:  # Iterate in batches over the training dataset with the batch_size set during the data loader 
         DT = []#A list to save the loaded graphs 
         for i in range(len(data)): #We need to load the actual data from the files so we loop through the batch and load those files   
             File = data[i]#File name 
             with ZipFile('/Dataset/{}.zip'.format(File), 'r') as f:#Since the files are saved in zip files they need to be extracted from being zipped whicvh we perform here 
                    f.extractall('/Dataset')
              D = torch.load("Dataset/{}.pt".format(File))#Loads the actual graph object 
              DT.append(D)
              os.remove('/Dataset/{}.pt'.format(File))#Removes the .pt file to ensure that the system does not run out of space
             
         out = model(DT.x, DT.edge_index, DT.batch)  # Perform a single forward pass with the model where access the edge index and features of the graphs 
         loss = criterion(out, DT.y)  # Compute the loss of the current network 
         loss.backward()  # Derive gradients to be used to update the model 
         optimizer.step()  # Update parameters based on gradients using Adam for this step
         optimizer.zero_grad()  # Clears the gradients and prepares for the next iteration 

def test(loader):#This is used for evaluating the performance on the model on the validation dataset and also the training dataset 
     model.eval()#This activates the evaluation mode for the model and tells it to simply evaluate rather than for example train 

     correct = 0
     
     for data in loader:  # Iterate in batches over the training/validation dataset.
          DT = []
          for i in range(len(data)):    
              File = data[i]
              with ZipFile('/Dataset/{}.zip'.format(File), 'r') as f:#Since the files are saved in zip files they need to be extracted from being zipped whicvh we perform here 
                    f.extractall('/Dataset')
              D = torch.load("Dataset/{}.pt".format(File))#Loads the actual graph object 
              DT.append(D)
              os.remove('/Dataset/{}.pt'.format(File))#Removes the .pt file to ensure that the system does not run out of space 
          out = model(DT.x, DT.edge_index, DT.batch)  
          pred = out.argmax(dim=1)  # Use the class with highest probability for the prediction.
          correct += int((pred == DT.y).sum())  # Check against labels of each graph and then add to correct number.
     return correct / len(loader.dataset)  # Returns the ratio of correct predictions over the total amount of data 



class GCN(torch.nn.Module):#This defines the graph convolutional model class that will be called when training 
    def __init__(self, hidden_channels):#Here we defined what type of layers with how many hidden channels should be involved in each layer
        super(GCN, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = GCNConv(4, hidden_channels)#The first graph convolutional layer, beyond 4 there are arguments that there is no real benefit to the model which needs to be tested a bit more 
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        
        self.linpre = Linear(hidden_channels, 32)#A linear layer that is essentially equivalent to a dense layer 
        
        self.lin = Linear(32, 4)

    def forward(self, x, edge_index, batch):#Defines how a forward pass in an iteration should occur 
        # 1. Obtain node embeddings using the convolutional layers 
        #x = x.float()
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        #x = x.relu()
        #x = self.conv4(x, edge_index)
        
        #x = x.relu()
        #x = self.conv4(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]#This reduces all of the output from the earlier layers to a flat set of parameters 

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.4, training=self.training)#Apply dropout to help with validation performance 
        x = self.linpre(x)
        x = self.lin(x)
        
        return x



    
model = GCN(hidden_channels=95)#Defines the model with a set number of hidden channels 

#Set optimizer to adam with a learning rate 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#0005  0.02

criterion = torch.nn.CrossEntropyLoss()#Cross Entropy loss function is used since it is a classification task 


for epoch in range(1, 51):#Here we set the number of epochs to perform the training over 
    train()#This tells the system to start training the network 
    
    train_acc = test(TrainD)#This tests the performance of the network on the training dataset 
    
    test_acc = test(TestD)#Tests the performance of the network on the training dataset 
    
    #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')#This can be used to be faster and only look at training performance simply comment out the test_acc
    
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')#Prints the result from both 
    
    
