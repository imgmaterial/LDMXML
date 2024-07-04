import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
class RNN_data_batch_generator(keras.utils.Sequence):#This is a generator that will load only the batch size number of files per iteration rather than loading alll of the data at once 
  
  def __init__(self, filenames, labels, batch_size,file_path, number_of_classes = 4) :
    self.filenames = filenames#The file names inputs 
    self.labels = labels#The labels input 
    self.batch_size = batch_size #The batch size once again
    self.number_of_classes = number_of_classes 
    self.file_path = file_path
    
  def __len__(self): #This gets the number of iterations that have to be performed to loop through all of the data and casts it as an integer 
    return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(int)

  def __getitem__(self, idx) :#This function actually loads the data 
    batch_x = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]#This gets the filenames in the batch and prepares for the data to be loaded 
    
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]#This deals with the labels of the data in the batch  
    
    Array = []

    
    for file_name in batch_x:#Loads each file with the filenames included 
        A = np.load(self.file_path + str(file_name))
        AA = A['arr_0']
        Array.append(AA)
    ADone = Array
    #To pass the data to a RNN network the arrays need to be the same length and so to ensure this the arrays are padded with zeros to ensure they all have a standardized length   
    ADone = tf.keras.preprocessing.sequence.pad_sequences(ADone, padding="post",maxlen = 454,dtype='float32')
    LL = []

    for i in batch_y:#Loop through the labels 
           JJ = i[0]-1 #Since our label goes from 1-4 we need to reduce it to 0-3 
           LL.append(JJ)
           
    YY = LL #Just renames it 
    YY = keras.utils.to_categorical(np.array(YY),num_classes=self.number_of_classes) #This changes the labels into an array with categories that can be handled by the model 
    print(YY.shape)
    return np.array(ADone), YY# This returns the loaded data and the labels
  
class CNN_data_batch_generator(keras.utils.Sequence):#This is a generator that will load only the batch size number of files per iteration rather than loading alll of the data at once 
  
  def __init__(self, filenames, labels, batch_size,file_path, number_of_classes = 4) :
    self.filenames = filenames#The file names inputs 
    self.labels = labels#The labels input 
    self.batch_size = batch_size #The batch size once again
    self.number_of_classes = number_of_classes 
    self.file_path = file_path
    
    
  def __len__(self): #This gets the number of iterations that have to be performed to loop through all of the data and casts it as an integer 
    return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(int)

  def __getitem__(self, idx) :#This function actually loads the data 
    batch_x = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]#This gets the filenames in the batch and prepares for the data to be loaded 
    
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]#This deals with the labels of the data in the batch  
    
    Array = []

    
    for file_name in batch_x:#This will loop through the files included in the batch 
        A = np.load(self.file_path + str(file_name))#This assumes that the data is some file that can be directly loaded using np.load
        AA = A['arr_0']#In an .npz file the array is saved under this header 
        Array.append(AA)

        
    ADone = np.array(Array)#Make an array of the loaded data 
    print("Printing the shape",ADone.shape[0])#Prints the shape 
    ADone = ADone.reshape(ADone.shape[0],34,7,450,1)  #This ensures the data has the right shape for this case the shape is consistent with a data set that includes both Ecal and Trigger Scintillator data, if considered just Ecal change this to ADone = ADone.reshape(ADone.shape[0],34,7,450,1) 
    LL = []

    for i in batch_y:#Loop through the labels 
           JJ = i[0]-1 #Since our label goes from 1-4 we need to reduce it to 0-3 
           LL.append(JJ)
           
    YY = LL #Just renames it 
    YY = keras.utils.to_categorical(np.array(YY),num_classes=self.number_of_classes) #This changes the labels into an array with categories that can be handled by the model 
    print(YY.shape)
    return ADone, YY# This returns the loaded data and the labels

 
def create_filename_label_batch(data_directory:str):
    files = os.walk(data_directory).__next__()[2]#This will walk around the directory and pick out any files it discovers there 
    m = len(files)#This is the amount of files located in the directory 
    print("Number of files ",m)
    filenames = []#This stores the filenames 
    labels = np.zeros((m, 1))#This creates an array full of zeros that will be used for the labels on the 
    filenames_counter = 0#This counts how many files there are
    for file in files:
        filenames.append(file)#Appends file from the directory 
        labels[filenames_counter, 0] = int(file[0]) #Gets the label from the first letter of the file (i.e. a file should be named something like 1e{} for this to work)
        filenames_counter = filenames_counter + 1  
    return filenames, labels
