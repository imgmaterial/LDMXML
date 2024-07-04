import pandas as pd
from DataExtractor import DataExtractor
import matplotlib.pyplot as plt
import numpy as np
csv_base_path = "/projects/hep/fs9/shared/ldmx/users/pa8701os/LDMXML/TestSamples/Lists/"

csv_path = ["LIST1.csv","LIST2.csv","LIST3.csv","LIST4.csv"]
data_path = ["/projects/hep/fs9/shared/ldmx/ldcs/gridftp/mc24/v14/4.0GeV/v3.3.x-1e-v14/","/projects/hep/fs9/shared/ldmx/ldcs/gridftp/mc24/v14/4.0GeV/v3.3.x-2e-v14/","/projects/hep/fs9/shared/ldmx/ldcs/gridftp/mc24/v14/4.0GeV/v3.3.x-3e-v14/","/projects/hep/fs9/shared/ldmx/ldcs/gridftp/mc24/v14/4.0GeV/v3.3.x-4e-v14/"]
output_path = "/projects/hep/fs9/shared/ldmx/users/pa8701os/LDMXML/TestSamples/EcalEnergyTspos/"
event_count = [1,2,3,4]
max_length = [155,250,400,450]

def ExtractDataFromCSV(path_to_csv:str, data_path:str, event_count:int, max_length:int, output_path:str):
    DF = pd.read_csv(path_to_csv) #LIST is a list of file names that contains the root files with the data, each one contained 10000 data points for this specific project. This file can be produced using terminal commands such as ls>list while in the directory where the files are 
    DF.columns = ['0']
    NR = len(DF.axes[0])#Just get the number of rows in the file 
    #DataExtractor([data_path + "{}".format(DF['0'][i]) for i in range(NR)],event_count,max_length,output_path=output_path).Interloper()#This calls the EcalID and Energy 
    #DataExtractor([data_path + "{}".format(DF['0'][i]) for i in range(NR)],event_count,max_length,output_path=output_path).BREM()#This calls the BREM function
    #DataExtractor([data_path + "{}".format(DF['0'][i]) for i in range(NR)],event_count,max_length, output_path=output_path).TSPOS()#This calls the Trig function
    #DataExtractor([data_path + "{}".format(DF['0'][i]) for i in range(NR)],event_count,max_length, output_path=output_path).EventNumber()
    #DataExtractor([data_path + "{}".format(DF['0'][i]) for i in range(NR)],event_count,max_length, output_path=output_path).AmountOfReadoutHits()
    DataExtractor([data_path + "{}".format(DF['0'][i]) for i in range(NR)],event_count,max_length, output_path=output_path).NumberOfStraightTracks()
def create_energy_plot(path_to_csv:str, data_path:str, event_count:int, max_length:int, output_path:str):
    DF = pd.read_csv(path_to_csv) #LIST is a list of file names that contains the root files with the data, each one contained 10000 data points for this specific project. This file can be produced using terminal commands such as ls>list while in the directory where the files are 
    DF.columns = ['0']
    NR = len(DF.axes[0])#Just get the number of rows in the file 
    data = DataExtractor([data_path + "{}".format(DF['0'][i]) for i in range(NR)],event_count,max_length, output_path=output_path).TotalEnergy()
    return data

def create_unified_energy_array():
    csv_base_path = "/projects/hep/fs9/shared/ldmx/users/pa8701os/LDMXML/TestSamples/Lists/"
    csv_path = ["LIST1.csv","LIST2.csv","LIST3.csv","LIST4.csv"]
    data_path = ["/projects/hep/fs9/shared/ldmx/ldcs/gridftp/mc24/v14/4.0GeV/v3.3.x-1e-v14/","/projects/hep/fs9/shared/ldmx/ldcs/gridftp/mc24/v14/4.0GeV/v3.3.x-2e-v14/","/projects/hep/fs9/shared/ldmx/ldcs/gridftp/mc24/v14/4.0GeV/v3.3.x-3e-v14/","/projects/hep/fs9/shared/ldmx/ldcs/gridftp/mc24/v14/4.0GeV/v3.3.x-4e-v14/"]
    output_path = "/projects/hep/fs9/shared/ldmx/users/pa8701os/LDMXML/TestSamples/EcalEnergyTspos/"
    event_count = [1,2,3,4]
    max_length = [155,250,400,450]
    unified_list = []
    for i in range(4):
        ExtractDataFromCSV(csv_base_path + csv_path[i],data_path[i],event_count[i],max_length[i], output_path)
        data = create_energy_plot(csv_base_path + csv_path[i],data_path[i],event_count[i],max_length[i], output_path)
        unified_list.append(data)
    un_array = np.concatenate(unified_list, axis=0)
    return(un_array)

def create_readout_hits_plot(path_to_csv:str, data_path:str, event_count:int, max_length:int, output_path:str):
    DF = pd.read_csv(path_to_csv) #LIST is a list of file names that contains the root files with the data, each one contained 10000 data points for this specific project. This file can be produced using terminal commands such as ls>list while in the directory where the files are 
    DF.columns = ['0']
    NR = len(DF.axes[0])#Just get the number of rows in the file 
    data = DataExtractor([data_path + "{}".format(DF['0'][i]) for i in range(NR)],event_count,max_length, output_path=output_path).AmountOfReadoutHits()
    return data

#print(create_unified_energy_array())

for i in range(4):
    ExtractDataFromCSV(csv_base_path + csv_path[i],data_path[i],event_count[i],max_length[i], output_path)
    #data = create_readout_hits_plot(csv_base_path + csv_path[i],data_path[i],event_count[i],max_length[i], output_path)
    #n_bins = 370
    #plt.hist(data, bins = n_bins,alpha = 0.5,  range=(30,400))
