import pandas as pd
import numpy as np

from libDetDescr import EcalID, HcalID,EcalTriggerID

def RNN_Ecal_1e(ecal_path:str, energy_path:str,output_path:str, n:int = 1000, nc:int = 0):

    readerC = pd.read_csv(ecal_path, chunksize=n)

    for chunkC in readerC:
        nE = 0
        NRR = chunkC.shape[0]
        NC = chunkC.shape[1]
        readerE = pd.read_csv(energy_path, chunksize=n)
        for chunkE in readerE:
            if nE<nc:
                nE = nE+1
                continue
            elif nE>nc:
                break
            elif nE==nc:
                if nc == 0:
                    for j in range(n):
                        B = np.zeros(shape = (155))#The array in this case is simply a one dimensional array with a length consistent with the maximum number of hits in a 1 electron event, they size of each array needs to be constant even if the size of the event is shorter in order for the RNN to be able to deal with them while training. Of course when loading the data later the array will be padded in length by adding zeros that will be ignored by the network while training.
                        for i in range(155):#155 was the highest in all the events considered but if you're working with new data might have to change this number
                            D = chunkC['{}'.format(i)][j]
                            if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                DD = int(D)
                                C = EcalID(DD).cell()
                                M = EcalID(DD).module()
                                L = EcalID(DD).layer()
                                B[i] = (L)*10000000+1000000*(M)+1000*C+chunkE['{}'.format(i)][j]#Each hit is turned into a value using this formula that uses the Energy, Cell, Module, and Layer numbers. This formula is not absolute and is completely arbitrary, it did however have some sucess so modifications or similar versions are at least worth considering
                        B = np.array(B)
                        np.savez_compressed(output_path + '1e{}.npz'.format(nc), B)#Saves the array to a file named for example 1e1.npz in the NPZ file format.
                elif NRR <n:
                    for j in range(nc*n,nc*n+NRR):
                                B = np.zeros(shape = (155))
                                for i in range(155):
                                        D = chunkC['{}'.format(i)][j]
                                        if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                            DD = int(D)
                                            C = EcalID(DD).cell()
                                            M = EcalID(DD).module()
                                            L = EcalID(DD).layer()
                                            B[i] = (L)*10000000+1000000*(M)+1000*C+chunkE['{}'.format(i)][j]
                                B = np.array(B)
                                np.savez_compressed(output_path+'1e{}.npz'.format(nc), B)
                else:
                    for j in range(nc*NRR,(nc+1)*NRR):
                                B = np.zeros(shape = (155))
                                for i in range(155):
                                        D = chunkC['{}'.format(i)][j]
                                        if pd.isna(D) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                                            DD = int(D)
                                            C = EcalID(DD).cell()
                                            M = EcalID(DD).module()
                                            L = EcalID(DD).layer()
                                            B[i] = (L)*10000000+1000000*(M)+1000*C+chunkE['{}'.format(i)][j]
                                B = np.array(B)
                                np.savez_compressed(output_path + '1e{}.npz'.format(nc), B)
                nE = nE+1
        nc = nc+1
