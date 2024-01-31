#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import time
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Lambda, concatenate
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, RNN

from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
from tensorflow.keras import backend as K

from sklearn.metrics import *

import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical

import sys

import numpy as np 

import tensorflow as tf


import os
import shutil

train_dir = 'Output/Pre-Processing/RNN/ETS'#This is wherever you have saved the data that was produced after preprocessing that you want to feed to your network  

subdirs, dirs, files = os.walk('Output/Pre-Processing/RNN/ETS').__next__()#This will walk around the directory and pick out any files it discovers there 

m = len(files)#This is the amount of files located in the directory 

print(m)

filenames = []#This stores the filenames 

labels = np.zeros((m, 1))#This creates an array full of zeros that will be used for the labels on the 


filenames_counter = 0#This counts how many files there are
labels_counter = -1#This counts how many lables there are

for subdir, dirs, files in os.walk(train_dir):
    for file in files:
        filenames.append(file)#Appends file from the directory 
        labels[filenames_counter, 0] = int(file[0]) #Gets the label from the first letter of the file (i.e. a file should be named something like 1e{} for this to work)
        filenames_counter = filenames_counter + 1
    labels_counter = labels_counter+1
    
# For now just renames it, not really necessisary at this this stage 
y_labels_one_hot = labels


from sklearn.utils import shuffle #Shuffle does what it says and randomizes the file name order 

filenames_shuffled, y_labels_one_hot_shuffled = shuffle(filenames, y_labels_one_hot)


from sklearn.model_selection import train_test_split #This allows the splitting of the dataset into a training data set and a validation data set without having to do it by hand 


filenames_shuffled_numpy = np.array(filenames_shuffled)#This creates an array of the file names 

#This splits the data into a training and validation set with the test size specifying how much of the data is in the validation data set 
X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
    filenames_shuffled_numpy, y_labels_one_hot_shuffled, test_size=0.3, random_state=1)

#Just prints the size of the different datasets as a way to double check 
print(X_train_filenames.shape)
print(y_train.shape)    
print(X_val_filenames.shape)   
print(y_val.shape) 

#The batch_size specifies how large the minibatch size is going to be
batch_size = 32
#The number of categories that will be used 
NumberClasses = 4  


class My_Custom_Generator(keras.utils.Sequence):#This is a generator that will load only the batch size number of files per iteration rather than loading alll of the data at once 
  
  def __init__(self, filenames, labels, batch_size) :
    self.filenames = filenames#The file names inputs 
    self.labels = labels#The labels input 
    self.batch_size = batch_size #The batch size once again 
    
    
  def __len__(self): #This gets the number of iterations that have to be performed to loop through all of the data and casts it as an integer 
    return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(int)

  def __getitem__(self, idx) :#This function actually loads the data 
    batch_x = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]#This gets the filenames in the batch and prepares for the data to be loaded 
    
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]#This deals with the labels of the data in the batch  
    
    Array = []

    
    for file_name in batch_x:#Loads each file with the filenames included 
        A = np.load('Output/Pre-Processing/RNN/ETS/' + str(file_name))
        AA = A['arr_0']
        Array.append(AA)
        #Lengths.append(len(AA))
    #print(len(Array), len(Array[0]), len(Array[2]))    
    ADone = Array
    #print(ADone.shape[0])
    #To pass the data to a RNN network the arrays need to be the same length and so to ensure this the arrays are padded with zeros to ensure they all have a standardized length   
    ADone = tf.keras.preprocessing.sequence.pad_sequences(ADone, padding="post",maxlen = 454,dtype='float32')
    LL = []

    for i in batch_y:#Loop through the labels 
           JJ = i[0]-1 #Since our label goes from 1-4 we need to reduce it to 0-3 
           LL.append(JJ)
           
    YY = LL #Just renames it 
    YY = to_categorical(np.array(YY),num_classes=NumberClasses) #This changes the labels into an array with categories that can be handled by the model 
    print(YY.shape)
    return np.array(ADone), YY# This returns the loaded data and the labels
    
    



my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train, batch_size) #Loads the generator for the training data set 

my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size) #Loads the generator for the validation data set 


from tensorflow.keras import layers #We use layers when loading the RNN model 

from tensorflow.keras import regularizers #Regularizers can be used to affect validation set performance 




#Model used for the Ecal and modified Trigger Scintillator 

masking_layer = layers.Masking() #The masking layer tells the network what it should ignore in our case we want it to ignore zeros but we start by intilizing it here 


inp = Input(shape=(454, 1)) #The input shape of our arrays that have all been standardized 


masking = keras.layers.Masking(mask_value=0)(inp)#Here we pass the input layer inp to the masking layer and tell it that it should ignore the mask values 0

lstm_1 = keras.layers.GRU(128, activation='tanh',return_sequences=True)(masking)#This is the first GRU layer and since return sequences is True this means that it can back propagate with the other GRU layers 


lstm_2 = keras.layers.GRU(32, activation='tanh',return_sequences=True,recurrent_dropout=0)(lstm_1)#This layer also has recurrent dropout added in the layer 




lstm_e = keras.layers.GRU(32, activation='tanh',return_sequences=True,recurrent_dropout=0)(lstm_2)



lstm_3 = keras.layers.GRU(64, activation='tanh',recurrent_dropout=0)(lstm_e)



Dense1 = keras.layers.Dense(64,'relu')(lstm_3)#Adds a dense layer of interconnected nodes 



Dense2 = keras.layers.Dense(32,'relu')(Dense1)


Drop = keras.layers.Dropout(0)(Dense2)#A final dropout layer used to control validation set performance 


DenseF = keras.layers.Dense(4,'softmax')(Drop)#A final output layer with the number of categories 


model = keras.models.Model(inputs=inp, outputs=DenseF) #This initilizes a model object that uses the input layer inp and the final dense layer DenseF as output  

adam = Adam(learning_rate= 0.0003) #Defines the optimizer with the learning rate 

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics = ['accuracy'])#This compiles the model with the Categorical cross entropy loss function and adam optimizer 

#This starts the training of the network with only two iterations (epochs) being used with the RNNM saving the history of the network as well.
RNNM = model.fit(my_training_batch_generator,epochs = 50,verbose = 1,validation_data = my_validation_batch_generator) #,callbacks=[EarlyStopping(patience=15)])

model.summary()

model.save("ModelNameHere")#Saves the model and can resume training after stopping 

plt.figure()
plt.ylabel('Loss / Accuracy')
plt.xlabel('Epoch')
for k in RNNM.history.keys():#Can be used to plot the history of the network performance 
    plt.plot(RNNM.history[k], label = k) 
plt.legend(loc='best')
plt.savefig("ModelHistory.png")
plt.show()





