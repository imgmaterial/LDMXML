import pandas as pd
import numpy as np
from libDetDescr import EcalID, HcalID,EcalTriggerID

def RNN_Ecal_3e(ecal_path:str, energy_path:str,output_path:str, n:int = 1000, nc:int = 0):
    readerC = pd.read_csv(ecal_path, chunksize=n)
    for chunkC in readerC:
        nE = 0
        NRR = chunkC.shape[0]
        NC = chunkC.shape[1]
        readerE = pd.read_csv(energy_path, chunksize=n)
        for chunkE in readerE:
            #nTS = 0
            if nE<nc:
                nE = nE+1
                continue
            elif nE>nc:
                break
            elif nE==nc:
                if nc == 0:
                    for j in range(n):
                        B = np.zeros(shape = (360))
                        for i in range(360):#155 was the highest in all the events considered but if you're working with new data might have to change this number
                            D = chunkC['{}'.format(i)][j]
                            if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                DD = int(D)
                                C = EcalID(DD).cell()
                                M = EcalID(DD).module()
                                L = EcalID(DD).layer()
                                B[i] = (L)*10000000+1000000*(M)+1000*C+chunkE['{}'.format(i)][j]
                        B = np.array(B)
                        np.savez_compressed(output_path + '3e{}.npz'.format(nc), B)#Saves the array to a file named for example 1e1.npz in the NPZ file format.
                elif NRR <n:
                    for j in range(nc*n,nc*n+NRR):
                                B = np.zeros(shape = (360))
                                for i in range(360):
                                        D = chunkC['{}'.format(i)][j]
                                        if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                            DD = int(D)
                                            C = EcalID(DD).cell()
                                            M = EcalID(DD).module()
                                            L = EcalID(DD).layer()
                                            B[i] = (L)*10000000+1000000*(M)+1000*C+chunkE['{}'.format(i)][j]
                                B = np.array(B)
                                np.savez_compressed(output_path + '3e{}.npz'.format(nc), B)
                else:
                    for j in range(nc*NRR,(nc+1)*NRR):
                                B = np.zeros(shape = (360))
                                for i in range(360):
                                        D = chunkC['{}'.format(i)][j]
                                        if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                            DD = int(D)
                                            C = EcalID(DD).cell()
                                            M = EcalID(DD).module()
                                            L = EcalID(DD).layer()
                                            B[i] = (L)*10000000+1000000*(M)+1000*C+chunkE['{}'.format(i)][j]
                                B = np.array(B)
                                np.savez_compressed(output_path + '3e{}.npz'.format(nc), B)
                nE = nE+1
        nc = nc+1
