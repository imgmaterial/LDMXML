import pandas as pd
import numpy as np

n = 1000

nc = 0

nE = 0

nTS = 0

import libDetDescr

from libDetDescr import EcalID, HcalID,EcalTriggerID


readerC = pd.read_csv("/Directory/EcalIDFile1.csv", chunksize=n) #Divides some csv file containing the EcalIDs into units of size n, if the file is small enough then can instead simply load the entire file


for chunkC in readerC:
    nE = 0
    NRR = chunkC.shape[0]
    NC = chunkC.shape[1]
    readerE = pd.read_csv("/Directory/Energy1.csv", chunksize=n)#Load the energy file into chunks as well, again if the energy and EcalID file are both small enough to be kept in RAM there is no real need but that is highly unlikely for bigger datasets.
    for chunkE in readerE:
        nTS = 0
        if nE<nc:#The chunking is done with an iterator that does not act as a list so to ensure that chunkC and chunkE contain the same datapoints we count nE to ensure it becomes equal with nc that counts which iteration chunkC is on.
            nE = nE+1
            continue
        elif nE>nc:
            break
        elif nE==nc:#When they are equal we continue with the preprocessing
            readerTS = pd.read_csv("TSPOS1.csv", chunksize=n)#Once again assume that the TS file is too large
            for chunkTS in readerTS:
                if nTS<nc:#Now need to match all three numbers nc, nE, and nTS to ensure that the right section of data is being used
                    nTS = nTS+1
                    continue
                elif nTS>nc:
                    break
                elif nTS == nc and nTS == nE:
                    if nc==0:
                        for j in range(n):
                            B = np.zeros((35,8,450))#35 and 8 needed rather than 34 and 7 since added the extra TS data
                            for i in range(156):#This is one way of looping through the data by adding an extra iteration that is the TS data
                                if i == 0:
                                        B[0][0][int(chunkTS['{}'.format('y')][j])] = 1#The TS data is saved in columns 'x' and 'y' for 1 electron events while for 2 electron: x1, x2, y1, y2 and so on for the others. For higher multiplicity events this number scales so for 2 electron events would have without TS 250 but with 252 see 2eCNNETS.py
                                else:
                                    D = chunkC['{}'.format(i-1)][j]#Since initiated i and skipped 0 need to ensure all of the data in the event so reduce i by one.
                                    if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                            DD = int(D)
                                            C = EcalID(DD).cell()
                                            M = EcalID(DD).module()
                                            L = EcalID(DD).layer()
                                            B[L+1][M][C] = chunkE['{}'.format(i)][j]#L+1 rather than L since the first layer is reserved for the TS data point
                            B = np.array(B)
                            np.savez_compressed('/Directory/CNN/ETS/1e{}.npz'.format(nc), B)#Saves the array to a file named for example 1e1.npz in the NPZ file format.
                    elif NRR <n:
                        for j in range(nc*n,nc*n+NRR):
                            B = np.zeros((35,8,450))
                            for i in range(156):
                                if i == 0:
                                        B[0][0][int(chunkTS['{}'.format('y')][j])] = 1
                                else:
                                    D = chunkC['{}'.format(i-1)][j]
                                    if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                            DD = int(D)
                                            C = EcalID(DD).cell()
                                            M = EcalID(DD).module()
                                            L = EcalID(DD).layer()
                                            B[L+1][M][C] = chunkE['{}'.format(i)][j]
                            B = np.array(B)
                            np.savez_compressed('/Directory/CNN/ETS/1e{}.npz'.format(nc), B)
                    else:
                        for j in range(nc*NRR,(nc+1)*NRR):
                            B = np.zeros((35,8,450))
                            for i in range(156):
                                if i == 0:
                                    B[0][0][int(chunkTS['{}'.format('y')][j])] = 1
                                else:
                                    D = chunkC['{}'.format(i-1)][j]
                                    if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                            DD = int(D)
                                            C = EcalID(DD).cell()
                                            M = EcalID(DD).module()
                                            L = EcalID(DD).layer()
                                            B[L+1][M][C] = chunkE['{}'.format(i)][j]
                            B = np.array(B)
                            np.savez_compressed('/Directory/CNN/ETS/1e{}.npz'.format(nc), B)
                    nTS = nTS+1
            nE = nE+1
    nc = nc+1
