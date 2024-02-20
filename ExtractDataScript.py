import pandas as pd
from DataExtractor import DataExtractor
csv_path = ["Directory/LISTTEST1.csv"]
data_path = ["/projects/hep/fs9/shared/ldmx/ldcs/gridftp/mc23/v14/4.0GeV/v3.2.2-1e-v14/"]
output_path = "/projects/hep/fs9/shared/ldmx/users/pa8701os/LDMXML/single_file_test/"
event_count = [1,2,3,4]
max_length = [155,250,360,450]

def ExtractDataFromCSV(path_to_csv:str, data_path:str, event_count:int, max_length:int, output_path:str):
    DF = pd.read_csv(path_to_csv) #LIST is a list of file names that contains the root files with the data, each one contained 10000 data points for this specific project. This file can be produced using terminal commands such as ls>list while in the directory where the files are 
    DF.columns = ['0']
    NR = len(DF.axes[0])#Just get the number of rows in the file 
    DataExtractor([data_path + "{}".format(DF['0'][i]) for i in range(NR)],event_count,max_length,output_path=output_path).Interloper()#This calls the EcalID and Energy 
    DataExtractor([data_path + "{}".format(DF['0'][i]) for i in range(NR)],event_count,max_length,output_path=output_path).BREM()#This calls the BREM function
    DataExtractor([data_path + "{}".format(DF['0'][i]) for i in range(NR)],event_count,max_length, output_path=output_path).Trig()#This calls the Trig function 


ExtractDataFromCSV(csv_path[0],data_path[0],event_count[0],max_length[0], output_path)