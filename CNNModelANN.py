#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import time


import tensorflow as tf #This imports tensorflow which this model is built in 


from tensorflow import keras
from tensorflow.keras.models import Model, Sequential#Model and sequential can both be used when building the model 
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Activation
from tensorflow.keras.layers import Conv3D,Conv2D, MaxPooling2D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Lambda, concatenate

from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam #This imports a series of different optimizers that can be tested 
from tensorflow.keras import backend as K

from sklearn.metrics import *

import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical #To categorical is useful for dealing with tasks with 
from tensorflow.keras import layers 
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import EarlyStopping #Early stopping allows for stopping when the changes are small while training the model 

import os
import shutil

train_dir = 'DataDirectory'#This is wherever you have saved the data that was produced after preprocessing that you want to feed to your network  

subdirs, dirs, files = os.walk('DataDirectory').__next__()#This will walk around the directory and pick out any files it discovers there 

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
    return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)

  def __getitem__(self, idx) :#This function actually loads the data 
    batch_x = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]#This gets the filenames in the batch and prepares for the data to be loaded 
    
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]#This deals with the labels of the data in the batch  
    
    Array = []

    
    for file_name in batch_x:#This will loop through the files included in the batch 
        A = np.load('/DataDirectory/' + str(file_name))#This assumes that the data is some file that can be directly loaded using np.load
        AA = A['arr_0']#In an .npz file the array is saved under this header 
        Array.append(AA)

        
    ADone = np.array(Array)#Make an array of the loaded data 
    print(ADone.shape[0])#Prints the shape 
    ADone = ADone.reshape(ADone.shape[0],35,8,450,1)#This ensures the data has the right shape for this case the shape is consistent with a data set that includes both Ecal and Trigger Scintillator data, if considered just Ecal change this to ADone = ADone.reshape(ADone.shape[0],34,7,450,1) 
    LL = []

    for i in batch_y:#Loop through the labels 
           JJ = i[0]-1 #Since our label goes from 1-4 we need to reduce it to 0-3 
           LL.append(JJ)
           
    YY = LL #Just renames it 
    YY = to_categorical(np.array(YY),num_classes=NumberClasses) #This changes the labels into an array with categories that can be handled by the model 
    print(YY.shape)
    return np.array(ADone), YY# This returns the loaded data and the labels 
    
    



my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train, batch_size)#This loads a training batch generator 

my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size)#This loads a validation batch generator 


from tensorflow.keras.layers import BatchNormalization #BatchNormalization can be used to normalize the input arrays which can make it easier for the network to deal with the arrays especially if the data has a lot of large or small numbers 

from tensorflow.keras import regularizers #Regulizers help with validation performance, you can combine l1l2() or use them seperately 


#The Model here was Used for the Ecal and Trigger Scintillator Dataset

model = Sequential()#Sequential allows for the building of models by adding layers in order 


model.add(layers.Conv3D(64,(2,2,2),activation='relu',input_shape=(35,8,450,1)))#This adds a 3D convolutional layer 
model.add(BatchNormalization())#This adds a batch normalization layer with no specified axis 
model.add(layers.MaxPooling3D((2,2,2))) #Max Pooling layer in 3D



model.add(layers.Conv3D(64,(2,2,2),activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPooling3D((1,1,2))) 



model.add(layers.Conv3D(64,(2,1,2),activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPooling3D((1,1,3))) 



model.add(layers.Conv3D(64,(2,2,2),activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPooling3D((2,1,5))) 



model.add(layers.Conv3D(64,(2,1,2),activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPooling3D((2,1,3))) 

model.add(tf.keras.layers.Flatten())#This layer flattens the data

model.add(layers.Dense(128,'tanh',kernel_regularizer=regularizers.L1(l1=1e-3))) #This adds an L1 regularizer in a dense layer thsat is just a series of connected nodes 

model.add(Dropout(0.6))#This adds a dropout layer that aids validation set performance 

model.add(layers.Dense(NumberClasses,'softmax'))#Adds a final output layer using softmax 

model.summary()#This prints a summary of the model 


learning_rate = 0.005



#Loss Function: CategoricalCrossentropy

model.compile(Adam(learning_rate),tf.keras.losses.CategoricalCrossentropy(),['accuracy'])#This compiles the model with an Adam optimizer and the Categorial Crossentropy loss function 

#Estimator is just a name for the results from our model it can be named anything, model.fit() initiates the training of the network and it will loop over the data. This will be performed 15 times as epochs  = 15, early stopping can be added to the model as well 
Estimator = model.fit(my_training_batch_generator,epochs = 15,verbose = 1,validation_data = my_validation_batch_generator) #,callbacks=[EarlyStopping(patience=15)])

model.save("ModelNameHere")#This saves the model, you can train and then stop and then resume the training just have to reload the model 

plt.figure()
plt.ylabel('Loss / Accuracy')
plt.xlabel('Epoch')
for k in Estimator.history.keys():#This makes a plot of the overall performance of the model over the number of epochs 
    plt.plot(Estimator.history[k], label = k) 
plt.legend(loc='best')
plt.show()
plt.savefig("ModelHistory.png")



#Load the model for extra work by using keras.models.load_model

#reconstruct = keras.models.load_model("ModelNameHere")









