import pandas as pd
import numpy as np
from lib.pre_processing_help import process_coordinates, process_value
from libDetDescr import EcalID, HcalID,EcalTriggerID

def RNN_ETS(event_multiplicity:int,ecal_path:str, energy_path:str,tspos_path:str, output_path:str, chunk_size:int = 1000, max_hits:int = 450):
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
        number_of_tspos_columns = len(chunkTspos.columns)
        amount_of_coordinates = int(number_of_tspos_columns/2)
        for j in range(indexEcal*chunk_size,indexEcal*chunk_size+number_of_rows):
            y_list = [process_value(chunkTspos['Y{}'.format(str(i))][j]) for i in range(1, amount_of_coordinates + 1)]
            x_list = [process_value(chunkTspos['X{}'.format(str(i))][j]) for i in range(1, amount_of_coordinates + 1)]
            coordinate_tuple = process_coordinates((x_list, y_list),amount_of_coordinates)
            y_list = np.unique(coordinate_tuple[1])
            hit_amount = len(y_list)
            B = np.zeros(shape = (max_hits + hit_amount))
            for i in range(hit_amount):
                B[i] = (y_list[i]*1000+1)
            for i in range(hit_amount, max_hits + hit_amount):
                D = chunkEcal['{}'.format(i-hit_amount)][j]
                if pd.isna(D) == False and pd.isna(chunkEnergy['{}'.format(i)][j])==False:
                    DD = int(D)
                    C = EcalID(DD).cell()
                    M = EcalID(DD).module()
                    L = EcalID(DD).layer()
                    B[i] = (L)*10000000+1000000*(M)+1000*C+chunkEnergy['{}'.format(i)][j]
            np.savez_compressed(output_path+'{}e{:03}_{:03}.npz'.format(event_multiplicity,indexEcal,j - indexEcal*chunk_size), B)