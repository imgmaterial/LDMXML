import pandas as pd
import numpy as np
from libDetDescr import EcalID, HcalID,EcalTriggerID

def CNN_Ecal_1e(ecal_path:str, energy_path:str,output_path:str, n:int = 1000, nc:int = 0):
    readerC = pd.read_csv(ecal_path, chunksize=n)#Divides some csv file containing the EcalIDs into units of size n, if the file is small enough then can instead simply load the entire file
    for chunkC in readerC:
        nE = 0
        NRR = chunkC.shape[0]
        NC = chunkC.shape[1]
        readerE = pd.read_csv(energy_path, chunksize=n)#Load the energy file into chunks as well, again if the energy and EcalID file are both small enough to be kept in RAM there is no real need but that is highly unlikely for bigger datasets.
        for chunkE in readerE:
            if nE<nc: #The chunking is done with an iterator that does not act as a list so to ensure that chunkC and chunkE contain the same datapoints we count nE to ensure it becomes equal with nc that counts which iteration chunkC is on.
                nE = nE+1
                continue
            elif nE>nc:
                break
            elif nE==nc:#When they are equal we continue with the preprocessing
                if nc == 0:
                    for j in range(n):
                        B = np.zeros((34,7,450))#The size of the Array for the Ecal Dataset, changing the size here should not break anything assuming the training are changed as well
                        for i in range(155):#155 was the highest in all the events considered but if you're working with new data might have to change this number 
                            D = chunkC['{}'.format(i)][j]
                            if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:#The EcalID and Energy files are filled data and then NAN values, we want to remove the NAN so we ignore them
                                DD = int(D)
                                C = EcalID(DD).cell()
                                M = EcalID(DD).module()
                                L = EcalID(DD).layer()
                                B[L][M][C] = chunkE['{}'.format(i)][j] #Each point where there is data in the array is set equal to the corresponding energy 
                        B = np.array(B)
                        np.savez_compressed(output_path + '1e{}.npz'.format(nc), B)#Saves the array to a file named for example 1e1.npz in the NPZ file format, if other file formats are used be sure to change the loading in the training files to be compliant.
                elif NRR <n:#This deals with the end of the csv file where the chunks can have a length smaller than n
                    for j in range(nc*n,nc*n+NRR):
                                B = np.zeros((34,7,450))
                                for i in range(155):
                                        D = chunkC['{}'.format(i)][j]
                                        if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                            DD = int(D)
                                            C = EcalID(DD).cell()
                                            M = EcalID(DD).module()
                                            L = EcalID(DD).layer()
                                            B[L][M][C] = chunkE['{}'.format(i)][j]
                                B = np.array(B)
                                np.savez_compressed(output_path + '1e{}.npz'.format(nc), B)
                else:
                    for j in range(nc*NRR,(nc+1)*NRR):
                                B = np.zeros((34,7,450))
                                for i in range(155):
                                        D = chunkC['{}'.format(i)][j]
                                        if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                            DD = int(D)
                                            C = EcalID(DD).cell()
                                            M = EcalID(DD).module()
                                            L = EcalID(DD).layer()
                                            B[L][M][C] = chunkE['{}'.format(i)][j]
                                B = np.array(B)
                                np.savez_compressed(output_path + '1e{}.npz'.format(nc), B)
                nE = nE+1#Updates the number and exits the loop
        nc = nc+1
