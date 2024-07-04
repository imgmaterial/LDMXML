import lib.model_generator as gen
import numpy as np
from sklearn.utils import shuffle #Shuffle does what it says and randomizes the file name order 
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import time
import sys

#pooling = int(sys.argv[1])
training_directory = "/projects/hep/fs9/shared/ldmx/users/pa8701os/LDMXML/8mil_processing/CNNEcalTrigPooled/"
model_dir = "Models/CNNEcal8milTrigPooled_jacob"
checkpoint_dir = 'Checkpoints/CNNEcal8milPooled_jacob'


def TrainNaivePooledCNN(training_directory, model_dir, checkpoint_dir, pooling = 1):
    batch_size = 32
    number_of_classes = 4
    filenames, labels = gen.create_create_filename_label_batch(training_directory)

    filenames_shuffled, y_labels_one_hot_shuffled = shuffle(filenames, labels)
    filenames_shuffled_numpy = np.array(filenames_shuffled)
    X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
        filenames_shuffled_numpy, y_labels_one_hot_shuffled, test_size=0.3, random_state=1)

    my_training_batch_generator = gen.CNN_data_batch_generator(X_train_filenames, y_train, batch_size, training_directory, pooling_rate=pooling)#This loads a training batch generator 

    my_validation_batch_generator = gen.CNN_data_batch_generator(X_val_filenames, y_val, batch_size,training_directory, pooling_rate=pooling)

    model = gen.compile_CNN_model(input_shape=(34,7,int(450/pooling),1))

    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(filepath=checkpoint_dir,save_best_only=True, monitor='accuracy', mode='max', save_freq=1000, verbose=1)  
    Estimator = model.fit(my_training_batch_generator,epochs = 1,verbose = 1,validation_data = my_validation_batch_generator,callbacks=[checkpoint]) #,callbacks=[EarlyStopping(patience=15)])
    model.save(model_dir)

def TriggerPooledCNN(training_directory, model_dir, checkpoint_dir, pooling = 1):
    batch_size = 32
    number_of_classes = 4
    filenames, labels = gen.create_create_filename_label_batch(training_directory)

    filenames_shuffled, y_labels_one_hot_shuffled = shuffle(filenames, labels)
    filenames_shuffled_numpy = np.array(filenames_shuffled)
    X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
        filenames_shuffled_numpy, y_labels_one_hot_shuffled, test_size=0.3, random_state=1)

    my_training_batch_generator = gen.CNN_data_batch_generator(X_train_filenames, y_train, batch_size, training_directory, model_type=gen.ModelType.EcalTrigPooled)#This loads a training batch generator 

    my_validation_batch_generator = gen.CNN_data_batch_generator(X_val_filenames, y_val, batch_size,training_directory, model_type=gen.ModelType.EcalTrigPooled)

    model = gen.compile_CNN_model(input_shape=(34,7,48,1))

    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(filepath=checkpoint_dir,save_best_only=True, monitor='accuracy', mode='max', save_freq=1000, verbose=1)  
    Estimator = model.fit(my_training_batch_generator,epochs = 1,verbose = 1,validation_data = my_validation_batch_generator,callbacks=[checkpoint]) #,callbacks=[EarlyStopping(patience=15)])
    model.save(model_dir)

start_time = time.time()
print("Start time {}".format(start_time))
TriggerPooledCNN(training_directory, model_dir, checkpoint_dir, pooling=1)
end_time = time.time()
print("End time {}".format(end_time))
execution_time = end_time - start_time
print("Execution time {}".format(execution_time))
