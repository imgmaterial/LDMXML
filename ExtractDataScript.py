import pandas as pd
from DataExtractor import DataExtractor
csv_path = ["Directory/LIST1.csv","Directory/LIST2.csv","Directory/LIST3.csv","Directory/LIST4.csv"]
data_path = ["/projects/hep/fs9/shared/ldmx/ldcs/gridftp/mc23/v14/4.0GeV/v3.2.2-1e-v14/",
              "/projects/hep/fs9/shared/ldmx/ldcs/gridftp/mc23/v14/4.0GeV/v3.2.2-2e-v14/",
              "/projects/hep/fs9/shared/ldmx/ldcs/gridftp/mc23/v14/4.0GeV/v3.2.2-3e-v14/",
              "/projects/hep/fs9/shared/ldmx/ldcs/gridftp/mc23/v14/4.0GeV/v3.2.2-4e-v14/"]
event_count = [1,2,3,4]
max_length = [155,250,360,450]

def ExtractDataFromCSV(path_to_csv:str, data_path:str, event_count:int, max_length:int):
    DF = pd.read_csv(path_to_csv) #LIST is a list of file names that contains the root files with the data, each one contained 10000 data points for this specific project. This file can be produced using terminal commands such as ls>list while in the directory where the files are 
    DF.columns = ['0']
    NR = len(DF.axes[0])#Just get the number of rows in the file 
    DataExtractor([data_path + "{}".format(DF['0'][i]) for i in range(NR)],event_count,max_length).Interloper()#This calls the EcalID and Energy 
    DataExtractor([data_path + "{}".format(DF['0'][i]) for i in range(NR)],event_count,max_length).BREM()#This calls the BREM function
    DataExtractor([data_path + "{}".format(DF['0'][i]) for i in range(NR)],event_count,max_length).Trig()#This calls the Trig function 

for i in range(len(csv_path)):
    ExtractDataFromCSV(csv_path[i],data_path[i],event_count[i],max_length[i])


