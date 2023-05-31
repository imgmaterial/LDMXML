#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import tensorflow as tf
import time

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Activation
from tensorflow.keras.layers import Conv3D,Conv2D, MaxPooling2D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Lambda, concatenate
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, RNN

from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
from tensorflow.keras import backend as K

from sklearn.metrics import *

import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


import os
import shutil

train_dir = 'DataDirectory'
#dest_dir = '/content/all_images'
counter = 0



subdirs, dirs, files = os.walk('DataDirectory').__next__()
m = len(files)
print(m)

filenames = []
labels = np.zeros((m, 1))

import os
import shutil


filenames_counter = 0
labels_counter = -1

for subdir, dirs, files in os.walk(train_dir):
    for file in files:
        filenames.append(file)
        labels[filenames_counter, 0] = int(file[0])
        filenames_counter = filenames_counter + 1
    labels_counter = labels_counter+1
    
# One hot vector representation of labels
y_labels_one_hot = labels



# saving the y_labels_one_hot array as a .npy file
#np.save('y_labels_one_hotTEST.npy', y_labels_one_hot)



#print(y_labels_one_hot)

from sklearn.utils import shuffle

filenames_shuffled, y_labels_one_hot_shuffled = shuffle(filenames, y_labels_one_hot)


from sklearn.model_selection import train_test_split


filenames_shuffled_numpy = np.array(filenames_shuffled)

X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
    filenames_shuffled_numpy, y_labels_one_hot_shuffled, test_size=0.3, random_state=1)


print(X_train_filenames.shape)



print(y_train.shape)    
    
print(X_val_filenames.shape)   
print(y_val.shape) 


batch_size = 64


NumberClasses = 4  


class My_Custom_Generator(keras.utils.Sequence):
  
  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self): 
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    Array = []
    Lengths = []
    
    for file_name in batch_x:
        A = np.load('DataDirectory/' + str(file_name))
        AA = A['arr_0']
        Array.append(AA)
        Lengths.append(len(AA))
        
    ADone = np.array(Array)
    print(ADone.shape[0])
    ADone = ADone.reshape(ADone.shape[0],35,8,450,1)
    LL = []

    for i in batch_y:
           JJ = i[0]-1 
           LL.append(JJ)
           
    YY = LL 
    YY = to_categorical(np.array(YY),num_classes=NumberClasses) 
    print(YY.shape)
    return np.array(ADone), YY
    
    



my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train, batch_size)

my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size)



from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers



#Model used for Ecal and Trigger Scintillator data

model = Sequential()
#model.add(ConvLSTM2D(filters = 4, kernel_size = (3, 3), activation = 'tanh',data_format = "channels_last",recurrent_dropout=0.2, return_sequences=True, input_shape = (34,1, 71, 1)))
model.add(ConvLSTM2D(filters = 8, kernel_size = (2,2), activation = 'tanh',data_format = "channels_last",recurrent_dropout=0.2 ,return_sequences=True,input_shape = (35,8, 450, 1)))
model.add(BatchNormalization())

model.add(tf.keras.layers.MaxPooling3D(pool_size=(15,2, 25)))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters = 4, kernel_size = (2, 2), activation = 'tanh',data_format = "channels_last",recurrent_dropout=0.2 ,return_sequences=True,input_shape = (35,8, 450, 1)))
model.add(BatchNormalization())

model.add(tf.keras.layers.MaxPooling3D(pool_size=(1,1, 5)))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters = 8, kernel_size = (2, 2), activation = 'tanh',data_format = "channels_last",recurrent_dropout=0.2 ,return_sequences=True,input_shape = (35,8, 450, 1)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(128,'tanh'))

model.add(Dropout(0.5))
model.add(Dense(NumberClasses,'softmax',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))


model.summary()



learning_rate = 0.005



#Loss Function: CategoricalCrossentropy

model.compile(Adam(learning_rate),tf.keras.losses.CategoricalCrossentropy(),['accuracy'])

Estimator = model.fit(my_training_batch_generator,epochs = 15,verbose = 1,validation_data = my_validation_batch_generator) #,callbacks=[EarlyStopping(patience=15)])

model.save("ModelNameHere")

plt.figure()
plt.ylabel('Loss / Accuracy')
plt.xlabel('Epoch')
for k in Estimator.history.keys():
    plt.plot(Estimator.history[k], label = k) 
plt.legend(loc='best')
plt.show()
plt.savefig("ModelHistory.png")











