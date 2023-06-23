import torch

from torch_geometric.utils import from_networkx

import pandas as pd

import numpy as np

import networkx as nx

import os

from zipfile import ZipFile

import zipfile

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
                            C = []
                            M = []
                            L = []
                            E = []
                            for i in range(252):
                                if i == 0:
                                    Y1 = int(chunkTS['{}'.format('y1')][j])
                                    Y2 = int(chunkTS['{}'.format('y2')][j])
                                                                        
                                    if Y1 == Y2:
                                        C.append((int(chunkTS['{}'.format('y1')][j])))
                                        M.append(0)
                                        L.append(0)
                                        E.append(1)
                                    else:
                                        C.append((int(chunkTS['{}'.format('y1')][j])))
                                        M.append(0)
                                        L.append(0)
                                        E.append(1)
                                                                        
                                        C.append((int(chunkTS['{}'.format('y2')][j])))
                                        M.append(0)
                                        L.append(0)
                                        E.append(1)
                                elif i==1:
                                    continue
                                else:
                                    if pd.isna(chunkC['{}'.format(i-2)][j]) == False and pd.isna(chunkE['{}'.format(i-2)][j])==False:
                                        D = int(chunkC['{}'.format(i-2)][j])
                                        C.append(EcalID(D).cell())
                                        M.append(EcalID(D).module())
                                        L.append(EcalID(D).layer() + 1)
                                        E.append(chunkE['{}'.format(i-2)][j])
                            MNN = nx.Graph()
                            MNN.add_nodes_from([(k,{"pos":(float(C[k]),float(M[k]),float(L[k])),"x":[float(E[k]),float(C[k]),float(M[k]),float(L[k])]   }) for k in range(len(C))] )
                            Edges = nx.geometric_edges(MNN, radius=40)
                            MNN.add_edges_from(Edges)
                            MNN = from_networkx(MNN)
                            MNN["y"] = 1
                            torch.save(MNN,'/Directory/GNN/ETS/2e{}.pt'.format(nc))
                            with zipfile.ZipFile('/Directory/GNN/ETS/2e{}.zip'.format(nc), 'w', zipfile.ZIP_DEFLATED) as f:
                                    f.write('/Directory/GNN/ETS/2e{}.pt'.format(nc))
                            os.remove('/Directory/GNN/ETS/2e{}.pt'.format(nc))
                    elif NRR <n:
                        for j in range(nc*n,nc*n+NRR):
                            C = []
                            M = []
                            L = []
                            E = []
                            for i in range(252):
                                if i == 0:
                                    Y1 = int(chunkTS['{}'.format('y1')][j])
                                    Y2 = int(chunkTS['{}'.format('y2')][j])
                                                                        
                                    if Y1 == Y2:
                                        C.append((int(chunkTS['{}'.format('y1')][j])))
                                        M.append(0)
                                        L.append(0)
                                        E.append(1)
                                    else:
                                        C.append((int(chunkTS['{}'.format('y1')][j])))
                                        M.append(0)
                                        L.append(0)
                                        E.append(1)
                                                                        
                                        C.append((int(chunkTS['{}'.format('y2')][j])))
                                        M.append(0)
                                        L.append(0)
                                        E.append(1)
                                elif i==1:
                                    continue
                                else:
                                    if pd.isna(chunkC['{}'.format(i-2)][j]) == False and pd.isna(chunkE['{}'.format(i-2)][j])==False:
                                        D = int(chunkC['{}'.format(i-2)][j])
                                        C.append(EcalID(D).cell())
                                        M.append(EcalID(D).module())
                                        L.append(EcalID(D).layer() + 1)
                                        E.append(chunkE['{}'.format(i-2)][j])
                            MNN = nx.Graph()
                            MNN.add_nodes_from([(k,{"pos":(float(C[k]),float(M[k]),float(L[k])),"x":[float(E[k]),float(C[k]),float(M[k]),float(L[k])]   }) for k in range(len(C))] )
                            Edges = nx.geometric_edges(MNN, radius=40)
                            MNN.add_edges_from(Edges)
                            MNN = from_networkx(MNN)
                            MNN["y"] = 1
                            torch.save(MNN,'/Directory/GNN/ETS/2e{}.pt'.format(nc))
                            with zipfile.ZipFile('/Directory/GNN/ETS/2e{}.zip'.format(nc), 'w', zipfile.ZIP_DEFLATED) as f:
                                    f.write('/Directory/GNN/ETS/2e{}.pt'.format(nc))
                            os.remove('/Directory/GNN/ETS/2e{}.pt'.format(nc))
                    else:
                        for j in range(nc*NRR,(nc+1)*NRR):
                            C = []
                            M = []
                            L = []
                            E = []
                            for i in range(252):
                                if i == 0:
                                    Y1 = int(chunkTS['{}'.format('y1')][j])
                                    Y2 = int(chunkTS['{}'.format('y2')][j])
                                                                        
                                    if Y1 == Y2:
                                        C.append((int(chunkTS['{}'.format('y1')][j])))
                                        M.append(0)
                                        L.append(0)
                                        E.append(1)
                                    else:
                                        C.append((int(chunkTS['{}'.format('y1')][j])))
                                        M.append(0)
                                        L.append(0)
                                        E.append(1)
                                                                        
                                        C.append((int(chunkTS['{}'.format('y2')][j])))
                                        M.append(0)
                                        L.append(0)
                                        E.append(1)
                                elif i==1:
                                    continue
                                else:
                                    if pd.isna(chunkC['{}'.format(i-2)][j]) == False and pd.isna(chunkE['{}'.format(i-2)][j])==False:
                                        D = int(chunkC['{}'.format(i-2)][j])
                                        C.append(EcalID(D).cell())
                                        M.append(EcalID(D).module())
                                        L.append(EcalID(D).layer() + 1)
                                        E.append(chunkE['{}'.format(i-2)][j])
                            MNN = nx.Graph()
                            MNN.add_nodes_from([(k,{"pos":(float(C[k]),float(M[k]),float(L[k])),"x":[float(E[k]),float(C[k]),float(M[k]),float(L[k])]   }) for k in range(len(C))] )
                            Edges = nx.geometric_edges(MNN, radius=40)
                            MNN.add_edges_from(Edges)
                            MNN = from_networkx(MNN)
                            MNN["y"] = 1
                            torch.save(MNN,'/Directory/GNN/ETS/2e{}.pt'.format(nc))
                            with zipfile.ZipFile('/Directory/GNN/ETS/2e{}.zip'.format(nc), 'w', zipfile.ZIP_DEFLATED) as f:
                                    f.write('/Directory/GNN/ETS/2e{}.pt'.format(nc))
                            os.remove('/Directory/GNN/ETS/2e{}.pt'.format(nc))
                    nTS = nTS+1
            nE = nE+1
    nc = nc+1
