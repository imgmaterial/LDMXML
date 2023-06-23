import pandas as pd
import numpy as np

n = 1000

nc = 0

nE = 0

nTS = 0

import libDetDescr

from libDetDescr import EcalID, HcalID,EcalTriggerID


readerC = pd.read_csv("/Directory/EcalIDFile2.csv", chunksize=n) #This assumes that the file containing the EcalIDs is too large to load into RAM if not then remove the chunking of the csv


for chunkC in readerC:#Will iterate over the chunks of the read csv file
    nE = 0
    NRR = chunkC.shape[0]
    NC = chunkC.shape[1]
    readerE = pd.read_csv("/Directory/Energy2.csv", chunksize=n)#
    for chunkE in readerE:
        nTS = 0
        if nE<nc:
            nE = nE+1
            continue
        elif nE>nc:
            break
        elif nE==nc:
            readerTS = pd.read_csv("TSPOS2.csv", chunksize=n)#Once again assuming that the TS file is too large
            for chunkTS in readerTS:
                if nTS<nc:
                    nTS = nTS+1
                    continue
                elif nTS>nc:
                    break
                elif nTS == nc and nTS == nE:
                    if nc==0:
                        for j in range(n):
                            B = np.zeros((35,8,450))
                            for i in range(251):
                                if i == 0:
                                        Y1 = int(chunkTS['{}'.format('y1')][j])
                                        Y2 = int(chunkTS['{}'.format('y2')][j])
                                        if Y1 == Y2:
                                            B[0][0][int(chunkTS['{}'.format('y1')][j])] = 1
                                        else:
                                            B[0][0][int(chunkTS['{}'.format('y1')][j])] = 1
                                            B[0][0][int(chunkTS['{}'.format('y2')][j])] = 1
                                else:
                                    D = chunkC['{}'.format(i-1)][j]
                                    if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                            DD = int(D)
                                            C = EcalID(DD).cell()
                                            M = EcalID(DD).module()
                                            L = EcalID(DD).layer()
                                            B[L+1][M][C] = chunkE['{}'.format(i)][j]
                            B = np.array(B)
                            np.savez_compressed('/Directory/CNN/ETS/2e{}.npz'.format(nc), B)
                    elif NRR <n:
                        for j in range(nc*n,nc*n+NRR):
                            B = np.zeros((35,8,450))
                            for i in range(251):
                                if i == 0:
                                        Y1 = int(chunkTS['{}'.format('y1')][j])
                                        Y2 = int(chunkTS['{}'.format('y2')][j])
                                        if Y1 == Y2:
                                            B[0][0][int(chunkTS['{}'.format('y1')][j])] = 1
                                        else:
                                            B[0][0][int(chunkTS['{}'.format('y1')][j])] = 1
                                            B[0][0][int(chunkTS['{}'.format('y2')][j])] = 1
                                else:
                                    D = chunkC['{}'.format(i-1)][j]
                                    if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                            DD = int(D)
                                            C = EcalID(DD).cell()
                                            M = EcalID(DD).module()
                                            L = EcalID(DD).layer()
                                            B[L+1][M][C] = chunkE['{}'.format(i)][j]
                            B = np.array(B)
                            np.savez_compressed('/Directory/CNN/ETS/2e{}.npz'.format(nc), B)
                    else:
                        for j in range(nc*NRR,(nc+1)*NRR):
                            B = np.zeros((35,8,450))
                            for i in range(251):
                                if i == 0:
                                        Y1 = int(chunkTS['{}'.format('y1')][j])
                                        Y2 = int(chunkTS['{}'.format('y2')][j])
                                        if Y1 == Y2:
                                            B[0][0][int(chunkTS['{}'.format('y1')][j])] = 1
                                        else:
                                            B[0][0][int(chunkTS['{}'.format('y1')][j])] = 1
                                            B[0][0][int(chunkTS['{}'.format('y2')][j])] = 1
                                else:
                                    D = chunkC['{}'.format(i-1)][j]
                                    if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                            DD = int(D)
                                            C = EcalID(DD).cell()
                                            M = EcalID(DD).module()
                                            L = EcalID(DD).layer()
                                            B[L+1][M][C] = chunkE['{}'.format(i)][j]
                            B = np.array(B)
                            np.savez_compressed('/Directory/CNN/ETS/2e{}.npz'.format(nc), B)
                    nTS = nTS+1
            nE = nE+1
    nc = nc+1
