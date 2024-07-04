from tensorflow import keras
from enum import Enum
import numpy as np
import os


class ModelType(Enum):
    Ecal = 1
    ETS = 2
    ETSX = 3
    EcalTrigPooled = 4

class RNN_data_batch_generator(keras.utils.Sequence):#This is a generator that will load only the batch size number of files per iteration rather than loading alll of the data at once 
  
  def __init__(self, filenames, labels, batch_size,file_path, number_of_classes = 4, array_size = 454) :
    self.filenames = filenames#The file names inputs 
    self.labels = labels#The labels input 
    self.batch_size = batch_size #The batch size once again
    self.number_of_classes = number_of_classes 
    self.file_path = file_path
    self.array_size = array_size

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
    ADone = keras.preprocessing.sequence.pad_sequences(ADone, padding="post",maxlen = self.array_size,dtype='float32')
    LL = []
    for i in batch_y:#Loop through the labels 
           JJ = i[0]-1 #Since our label goes from 1-4 we need to reduce it to 0-3 
           LL.append(JJ)   
    YY = LL #Just renames it 
    YY = keras.utils.to_categorical(np.array(YY),num_classes=self.number_of_classes) #This changes the labels into an array with categories that can be handled by the model 
    print(YY.shape)
    return np.array(ADone), YY# This returns the loaded data and the labels
  
class CNN_data_batch_generator(keras.utils.Sequence):#This is a generator that will load only the batch size number of files per iteration rather than loading alll of the data at once 
  
  def __init__(self, filenames, labels, batch_size,file_path, number_of_classes = 4, model_type = ModelType.Ecal,pooling_rate = 1):
    self.filenames = filenames#The file names inputs 
    self.labels = labels#The labels input 
    self.batch_size = batch_size #The batch size once again
    self.number_of_classes = number_of_classes 
    self.file_path = file_path
    self.model_type = model_type
    self.pooling_rate = pooling_rate
    
    
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
    if self.model_type == ModelType.Ecal:
        ADone = ADone.reshape(ADone.shape[0],34,7,int(450/self.pooling_rate),1)
    elif self.model_type == ModelType.EcalTrigPooled:
        ADone = ADone.reshape(ADone.shape[0],34,7,48,1)
    else:
        ADone = ADone.reshape(ADone.shape[0],35,8,450,1)
    LL = []
    for i in batch_y:#Loop through the labels 
           JJ = i[0]-1 #Since our label goes from 1-4 we need to reduce it to 0-3 
           LL.append(JJ)     
    YY = LL #Just renames it 
    YY = keras.utils.to_categorical(np.array(YY),num_classes=self.number_of_classes) #This changes the labels into an array with categories that can be handled by the model 
    return ADone, YY# This returns the loaded data and the labels

def create_create_filename_label_batch(data_directory:str):
    subdirs, dirs, files = os.walk(data_directory).__next__()
    m = len(files)
    filenames = []
    labels = np.zeros((m, 1))
    filenames_counter = 0
    labels_counter = -1
    for subdir, dirs, files in os.walk(data_directory):
        for file in files:
            filenames.append(file)
            labels[filenames_counter, 0] = int(file[0])
            filenames_counter = filenames_counter + 1
        labels_counter = labels_counter+1
    return filenames, labels


def standart_RNN_model_shapes(model_type:ModelType):
    if model_type == ModelType.Ecal:
        return [128,32,32,32,64,32,4]
    elif model_type == ModelType.ETS:
        return [128,32,32,64,64,32,4]
    elif model_type == ModelType.ETSX:
        return [128,32,32,128,64,32,4]

def standart_CNN_model_shapes(model_type:ModelType):
    if model_type == ModelType.Ecal:
        return [64,64,64,64,64,64]
    elif model_type == ModelType.ETS:
        return [64,64,64,64,64,128]
    elif model_type == ModelType.ETSX:
        return [64,64,64,64,64,128]

def compile_RNN_model(input_length:int = 500,learning_rate:float = 0.0001, droput_rate:float = 0, model_shape = standart_RNN_model_shapes(ModelType.Ecal)):
    masking_layer = keras.layers.Masking() #The masking layer tells the network what it should ignore in our case we want it to ignore zeros but we start by intilizing it here 
    inp = keras.layers.Input(shape=(input_length, 1)) #The input shape of our arrays that have all been standardized 
    masking = keras.layers.Masking(mask_value=0)(inp)#Here we pass the input layer inp to the masking layer and tell it that it should ignore the mask values 0
    lstm_1 = keras.layers.GRU(model_shape[0], activation='tanh',return_sequences=True)(masking)#This is the first GRU layer and since return sequences is True this means that it can back propagate with the other GRU layers 
    lstm_2 = keras.layers.GRU(model_shape[1], activation='tanh',return_sequences=True,recurrent_dropout=0)(lstm_1)#This layer also has recurrent dropout added in the layer 
    lstm_e = keras.layers.GRU(model_shape[2], activation='tanh',return_sequences=True,recurrent_dropout=0)(lstm_2)
    lstm_3 = keras.layers.GRU(model_shape[4], activation='tanh',recurrent_dropout=0)(lstm_e)
    Dense1 = keras.layers.Dense(model_shape[5],'relu')(lstm_3)#Adds a dense layer of interconnected nodes 
    Dense2 = keras.layers.Dense(model_shape[6],'relu')(Dense1)
    Drop = keras.layers.Dropout(droput_rate)(Dense2)#A final dropout layer used to control validation set performance 
    DenseF = keras.layers.Dense(model_shape[7],'softmax')(Drop)#A final output layer with the number of categories 
    model = keras.models.Model(inputs=inp, outputs=DenseF) #This initilizes a model object that uses the input layer inp and the final dense layer DenseF as output  
    adam = keras.optimizers.Adam(learning_rate= learning_rate) #Defines the optimizer with the learning rate 
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics = ['accuracy'])#This compiles the model with the Categorical cross entropy loss function and adam optimizer 
    return model

def compile_CNN_model(input_shape = (34,7,450,1), learning_rate = 0.005,number_of_classes:int = 4, dropout:float = 0.6, model_shape = standart_CNN_model_shapes(ModelType.Ecal)):
    model = keras.models.Sequential()#Sequential allows for the building of models by adding layers in order 
    model.add(keras.layers.Conv3D(model_shape[0],(2,2,2),padding = 'same', activation='relu',input_shape=input_shape))#This adds a 3D convolutional layer 
    model.add(keras.layers.BatchNormalization())#This adds a batch normalization layer with no specified axis 
    model.add(keras.layers.MaxPooling3D((2,2,2), padding = 'same')) #Max Pooling layer in 3D
    model.add(keras.layers.Conv3D(model_shape[1],(2,2,2),padding = 'same',activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling3D((1,1,2),padding = 'same')) 
    model.add(keras.layers.Conv3D(model_shape[2],(2,1,2),padding = 'same',activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling3D((1,1,3),padding = 'same')) 
    model.add(keras.layers.Conv3D(model_shape[3],(2,2,2),padding = 'same',activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling3D((2,1,5),padding = 'same')) 
    model.add(keras.layers.Conv3D(model_shape[4],(2,1,2),padding = 'same',activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling3D((2,1,3),padding = 'same')) 
    model.add(keras.layers.Flatten())#This layer flattens the data
    model.add(keras.layers.Dense(model_shape[5],'tanh',kernel_regularizer=keras.regularizers.L1(l1=1e-3))) #This adds an L1 regularizer in a dense layer thsat is just a series of connected nodes 
    model.add(keras.layers.Dropout(dropout))#This adds a dropout layer that aids validation set performance 
    model.add(keras.layers.Dense(number_of_classes,'softmax'))#Adds a final output layer using softmax 
    model.summary()#This prints a summary of the model 
    learning_rate = learning_rate
    model.compile(keras.optimizers.Adam(learning_rate),keras.losses.CategoricalCrossentropy(),['accuracy'])
    return model



