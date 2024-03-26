import pandas as pd
import numpy as np
from libDetDescr import EcalID, HcalID,EcalTriggerID
from lib.pre_processing_help import process_value, process_coordinates

def CNN_ETS(event_multiplicity:int,ecal_path:str, energy_path:str,tspos_path:str ,output_path:str, chunk_size:int = 1000, max_hits:int = 454):
    readerEcal = pd.read_csv(ecal_path, chunksize=chunk_size)
    for indexEcal, chunkEcal in enumerate(readerEcal):
        readerEnergy = pd.read_csv(energy_path,chunksize=chunk_size)
        readerTspos = pd.read_csv(tspos_path, chunksize=chunk_size)
        number_of_rows = chunkEcal.shape[0]
        for indexEnergy, _chunkEnergy in enumerate(readerEnergy): #Makes sure that we are looking at the corresponding events in ecal, energy and tspos
            if (indexEcal == indexEnergy):
                chunkEnergy = _chunkEnergy
                break
        for indexTspos, _chunkTspos in enumerate(readerTspos):
            if (indexEcal == indexTspos):
                chunkTspos = _chunkTspos
                break
        number_of_tspos_columns = len(chunkTspos.columns)
        amount_of_coordinates = int(number_of_tspos_columns/2)
        for j in range(indexEcal*chunk_size,indexEcal*chunk_size+number_of_rows): #Goes over events in the chunk
            y_list = [process_value(chunkTspos['Y{}'.format(str(i))][j]) for i in range(1, amount_of_coordinates + 1)]
            x_list = [process_value(chunkTspos['X{}'.format(str(i))][j]) for i in range(1, amount_of_coordinates + 1)]
            coordinate_tuple = process_coordinates((x_list, y_list),amount_of_coordinates)
            y_list = np.unique(coordinate_tuple[1])
            hit_amount = len(y_list)
            B = np.zeros((35,8,450))
            for hit in y_list:
                B[0][0][hit] = 1 #if a coordinate was hit changes a value of that coordinate from 0 to 1
            for i in range(1, max_hits + 1):
                D = chunkEcal['{}'.format(i-1)][j]
                if pd.isna(D) == False and pd.isna(chunkEnergy['{}'.format(i)][j])==False:
                    DD = int(D)
                    C = EcalID(DD).cell()
                    M = EcalID(DD).module()
                    L = EcalID(DD).layer()
                    B[L+1][M][C] = chunkEnergy['{}'.format(i)][j]
            np.savez_compressed(output_path+'{}e{:03}_{:03}.npz'.format(event_multiplicity,indexEcal,j - indexEcal*chunk_size), B)