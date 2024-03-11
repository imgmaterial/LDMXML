import pandas as pd
import numpy as np
from libDetDescr import EcalID, HcalID,EcalTriggerID

def RNN_Ecal(event_multiplicity:int,ecal_path:str, energy_path:str,tspos_path:str, output_path:str, chunk_size:int = 1000, max_hits:int = 454):
    readerEcal = pd.read_csv(ecal_path, chunksize=chunk_size)
    for indexEcal, chunkEcal in enumerate(readerEcal):
        readerEnergy = pd.read_csv(energy_path,chunksize=chunk_size)
        readerTspos = pd.read_csv(tspos_path, chunksize=chunk_size)
        number_of_rows = chunkEcal.shape[0]
        for indexEnergy, _chunkEnergy in enumerate(readerEnergy):
            if (indexEcal == indexEnergy):
                chunkEnergy = _chunkEnergy
                break
        for indexTspos, _chunkTspos in enumerate(readerTspos):
            if (indexEcal == indexTspos):
                chunkTspos = _chunkTspos
                break
        for j in range(indexEcal*chunk_size,indexEcal*chunk_size+number_of_rows):
            B = np.zeros(shape = (max_hits + event_multiplicity))
            y_list = [int(chunkTspos['Y{}'.format(str(i))][j]) for i in range(1, event_multiplicity + 1)]
            combine_like_entries(y_list)
            for i in range(event_multiplicity):
                B[i] = (y_list[i]*1000+1)
            for i in range(event_multiplicity, max_hits + event_multiplicity):
                D = chunkEcal['{}'.format(i)][j]
                if pd.isna(D) == False and pd.isna(chunkEnergy['{}'.format(i)][j])==False:
                    DD = int(D)
                    C = EcalID(DD).cell()
                    M = EcalID(DD).module()
                    L = EcalID(DD).layer()
                    B[i] = (L)*10000000+1000000*(M)+1000*C+chunkEnergy['{}'.format(i)][j]
            np.savez_compressed(output_path+'{}e{:03}_{:03}.npz'.format(event_multiplicity,indexEcal,j - indexEcal*chunk_size), B)
        
def combine_like_entries(y_list:list):
    for i in range(len(y_list)):
        for j in range(i+1,len(y_list)):
            if (y_list[i] == y_list[j]):
                y_list[j] = 0