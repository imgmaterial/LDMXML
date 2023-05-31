# LDMXML
This project was a Machine Learning project for my Master Thesis, the goal of which was to use Artificial Neural Networks (ANNs) to classify events from the Light Dark Matter eXperiment depending on the number of electrons in each event.  

The project uses simulated data generated via the the LDMX software (https://github.com/LDMX-Software/ldmx-sw).  The data is produced in a series of ROOT files and so Uproot is necessary for dealing with the data. The project is built in python 3.10 and is built with Tensorflow and Keras as well as PyTorch Geometric.  The data used in this project was generated as a series of ROOT files with 10000 events in each and the data extraction can be seen in DataExtractor.py.

Four different types of ANNs were trained for this task: A convolutional neural network (CNN), recurrent neural network (RNN), graph neural network (GNN), and a combined CNN and RNN neural network. The CNN, RNN, and combination networks were built using Tensorflow while the GNN were built using PyTorch Geometric. 

The different ANNs are designed to work with data from the Electromagnetic Calorimeter (Ecal) and the Trigger Scintillator. The data from the Equal are located in the EcalRecHits_sim parts of the ROOT file generated with the data while the Trigger Scintillator data is located in TrigScintScoringPlaneHits_sim. The models can be used with the Cartesian coordinate version of the data but is currently designed to use the Model, Layer, and Cell number of each hit in the event. This requires the use of the EcalID for each that can be converted to Module, Layer, and Cell ID using the EcalID() function from libDetDescr. The current preprocessing required for the different models included here can of course be changed and is something I would actively encourage to investigate the possible effects but currently they have the following forms:

CNN:

The CNN models use 3D convolutional layers and therefore require a 3D array as input. If only the Ecal data is used the arrat has the shape (34,7,450) while if the Ecal data is combined with the Trigger Scintillator data the shape is (35,8,450) the increased size comes from accounting for adding the Trigger Scintillator data for each hit. Worth testing is if by increasing the dimension size a deeper network can be created, for example by changing to (35,50,450) more layers could be used which might have some benefit on performance. The model can be seen in the CNNModelANN.py. 

RNN:

Each hit is converted into a value in a time series for each event. The RNN networks requires that the events are equal in length so they need to be padded to ensure they are all the same length. This padding is performed with zeros that are subsequently ignored by the network while performing the training. The largest length for the Ecal events was found to be 450 while for the Ecal and Trigger Scintillator 451 and 453 were used depending on the implementation. An example can be seen in RNNModelANN.py.


GNN: 

Each hit is treated as a node in a graph with features (energy, module, layer, cell) while the edges are calculated using geometric radius which is set to 40 for this data set. There is no real difference between the Ecal and the Ecal and Trigger Scintillator models as the number of hits are just different and there is no need for padding as the number of nodes is not a set number. An example can be seen in GNNModelANN.py. [The model included unfortunately does not work yet however this provides a basis to build of.]

Combined CNN and RNN:

The same array shape is used as for the CNN but the 34 or 35 now refers to the number of frames in the “video” that is being processed. Note that the array is reshaped into (34,7,450,1) and (35,8,450,1) as this becomes the expected shape for this type of network. An example can be seen in CRModelANN.py


Note on Data Generation: 
When generating the data a recommendation would be to save each event as a separate file. This allows for an easier time when using the generators later as changing the batch size becomes much easier to deal with. A generator only loads specific files at a time since trying to load all of the data at once overloads the RAM. 

Note on Training:
Start training with smaller datasets and find models that are viable at that size. Once a reasonable performance is reached for that smaller dataset expand to the larger and start to optimize for that total dataset. 


Required Modules:
- [ ] Uproot
- [ ] Pandas 
- [ ] Numpy 
- [ ] Tensorflow 
- [ ] Pytorch Geometric 
- [ ] Networkx
- [ ] Sklearn 


