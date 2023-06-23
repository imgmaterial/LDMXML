import pandas as pd
import numpy as np

n = 1000

nc = 0

nE = 0

nTS = 0

import libDetDescr

from libDetDescr import EcalID, HcalID,EcalTriggerID


readerC = pd.read_csv("/Directory/EcalIDFile4.csv", chunksize=n)


for chunkC in readerC:
    nE = 0
    NRR = chunkC.shape[0]
    NC = chunkC.shape[1]
    readerE = pd.read_csv("/Directory/Energy4.csv", chunksize=n)
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
                    B = np.zeros(shape = (450))
                    for i in range(450):#155 was the highest in all the events considered but if you're working with new data might have to change this number
                        D = chunkC['{}'.format(i)][j]
                        if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                            DD = int(D)
                            C = EcalID(DD).cell()
                            M = EcalID(DD).module()
                            L = EcalID(DD).layer()
                            B[i] = (L)*10000000+1000000*(M)+1000*C+chunkE['{}'.format(i)][j])
                    B = np.array(B)
                    np.savez_compressed('/Directory/RNN/Ecal/4e{}.npz'.format(nc), B)#Saves the array to a file named for example 1e1.npz in the NPZ file format.
            elif NRR <n:
                  for j in range(nc*n,nc*n+NRR):
                            B = np.zeros(shape = (450))
                            for i in range(450):
                                    D = chunkC['{}'.format(i)][j]
                                    if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                        DD = int(D)
                                        C = EcalID(DD).cell()
                                        M = EcalID(DD).module()
                                        L = EcalID(DD).layer()
                                        B[i] = (L)*10000000+1000000*(M)+1000*C+chunkE['{}'.format(i)][j])
                            B = np.array(B)
                            np.savez_compressed('/Directory/RNN/Ecal/4e{}.npz'.format(nc), B)
            else:
                for j in range(nc*NRR,(nc+1)*NRR):
                            B = np.zeros(shape = (450))
                            for i in range(450):
                                    D = chunkC['{}'.format(i)][j]
                                    if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                        DD = int(D)
                                        C = EcalID(DD).cell()
                                        M = EcalID(DD).module()
                                        L = EcalID(DD).layer()
                                        B[i] = (L)*10000000+1000000*(M)+1000*C+chunkE['{}'.format(i)][j])
                            B = np.array(B)
                            np.savez_compressed('/Directory/RNN/Ecal/4e{}.npz'.format(nc), B)
            nE = nE+1
    nc = nc+1
