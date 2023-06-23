import pandas as pd
import numpy as np

n = 1000

nc = 0

nE = 0

nTS = 0

import libDetDescr

from libDetDescr import EcalID, HcalID,EcalTriggerID


readerC = pd.read_csv("/Directory/EcalIDFile3.csv", chunksize=n)


for chunkC in readerC:
    nE = 0
    NRR = chunkC.shape[0]
    NC = chunkC.shape[1]
    readerE = pd.read_csv("/Directory/Energy3.csv", chunksize=n)
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
                    B = np.zeros((34,7,450))
                    for i in range(360):
                        D = chunkC['{}'.format(i)][j]
                        if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                            DD = int(D)
                            C = EcalID(DD).cell()
                            M = EcalID(DD).module()
                            L = EcalID(DD).layer()
                            B[L][M][C] = chunkE['{}'.format(i)][j]
                    B = np.array(B)
                    np.savez_compressed('/Directory/CNN/Ecal/3e{}.npz'.format(nc), B)
                  for j in range(nc*n,nc*n+NRR):
                            B = np.zeros((34,7,450))
                            for i in range(360):
                                    D = chunkC['{}'.format(i)][j]
                                    if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                        DD = int(D)
                                        C = EcalID(DD).cell()
                                        M = EcalID(DD).module()
                                        L = EcalID(DD).layer()
                                        B[L][M][C] = chunkE['{}'.format(i)][j]
                            B = np.array(B)
                            np.savez_compressed('/Directory/CNN/Ecal/3e{}.npz'.format(nc), B)
            else:
                for j in range(nc*NRR,(nc+1)*NRR):
                            B = np.zeros((34,7,450))
                            for i in range(360):
                                    D = chunkC['{}'.format(i)][j]
                                    if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                        DD = int(D)
                                        C = EcalID(DD).cell()
                                        M = EcalID(DD).module()
                                        L = EcalID(DD).layer()
                                        B[L][M][C] = chunkE['{}'.format(i)][j]
                            B = np.array(B)
                            np.savez_compressed('/Directory/CNN/Ecal/3e{}.npz'.format(nc), B)
            nE = nE+1
    nc = nc+1
