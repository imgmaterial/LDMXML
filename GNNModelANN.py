#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib as plt 


from torch_geometric.loader import DataLoader


from sklearn.model_selection import train_test_split


from torch_geometric.data import Data

#from torch_geometric import Dataset
from torch.utils.data import Dataset
from torch_geometric.nn.norm.batch_norm import BatchNorm

#from torchvision import transforms


import os 
import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GravNetConv, ChebConv
from torch_geometric.nn import global_mean_pool


filenames  = []


train_dir = 'Dataset'
k = 0

for subdir, dirs, files in os.walk(train_dir):
    #print(files)
    for file in files:
        #print(file[0])
        filenames.append(file)
        k = k+1
        if k==101:
             break
print(len(filenames))



TrainFiles, TestFiles = train_test_split(filenames,train_size=0.7,test_size=0.3)

print(len(TrainFiles))

print(len(TestFiles))

##train_data, test_data = random_split(torch_dataset, [1400, 600])


TrainD = DataLoader(TrainFiles, batch_size=64,shuffle=True)

TestD = DataLoader(TestFiles, batch_size=64,shuffle=False)







def train():
    model.train()

    for data in TrainD:  # Iterate in batches over the training dataset.
         DT = []
         for i in range(len(data)):    
             File = data[i]
             D = torch.load("Dataset/{}.pt".format(File))
             DT.append(D)
             
         out = model(DT.x, DT.edge_index, DT.batch)  # Perform a single forward pass.
         loss = criterion(out, DT.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     
     for data in loader:  # Iterate in batches over the training/validation dataset.
          DT = []
          for i in range(len(data)):    
              File = data[i]
              D = torch.load("Dataset/{}.pt".format(File))
              DT.append(D)
              
          out = model(DT.x, DT.edge_index, DT.batch)  
          pred = out.argmax(dim=1)  # Use the class with highest probability.
          correct += int((pred == DT.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.






import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GravNetConv, ChebConv#Different Graph convolutional layers: Experiment!
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = GCNConv(4, hidden_channels)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        
        self.linpre = Linear(hidden_channels, 32)
        
        self.lin = Linear(32, 4)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
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
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.linpre(x)
        x = self.lin(x)
        
        return x



    
model = GCN(hidden_channels=95)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#0005  0.02
criterion = torch.nn.CrossEntropyLoss()


for epoch in range(1, 51):
    train()
    
    train_acc = test(TrainD)
    
    test_acc = test(TestD)
    
    #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
    
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    
