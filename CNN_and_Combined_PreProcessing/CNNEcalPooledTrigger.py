import pandas as pd
import numpy as np
import lib.cells_to_trigger as trigger
from libDetDescr import EcalID, HcalID,EcalTriggerID

def CNN_Ecal(event_multiplicity:int,ecal_path:str, energy_path:str,output_path:str, chunk_size:int = 1000, max_hits:int = 454):
    readerEcal = pd.read_csv(ecal_path, chunksize=chunk_size)
    cell_to_trig = trigger.create_cells_to_trig_cells_map()
    for indexEcal, chunkEcal in enumerate(readerEcal):
        readerEnergy = pd.read_csv(energy_path,chunksize=chunk_size)
        number_of_rows = chunkEcal.shape[0]
        for indexEnergy, _chunkEnergy in enumerate(readerEnergy):
            if (indexEcal == indexEnergy):
                chunkEnergy = _chunkEnergy
                break
        for j in range(indexEcal*chunk_size,indexEcal*chunk_size+number_of_rows):
            B = np.zeros((34,7,48))
            for i in range(max_hits):
                D = chunkEcal['{}'.format(i)][j]
                if pd.isna(D) == False and pd.isna(chunkEnergy['{}'.format(i)][j])==False:
                    DD = int(D)
                    C = cell_to_trig[EcalID(DD).cell()]
                    M = EcalID(DD).module()
                    L = EcalID(DD).layer()
                    B[L][M][C] += chunkEnergy['{}'.format(i)][j]
            np.savez_compressed(output_path+'{}e{:03}_{:03}.npz'.format(event_multiplicity,indexEcal,j - indexEcal*chunk_size), B)
