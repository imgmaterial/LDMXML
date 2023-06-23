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


readerC = pd.read_csv("/Directory/EcalIDFile4.csv", chunksize=n)


for chunkC in readerC:
    nE = 0
    NRR = chunkC.shape[0]
    NC = chunkC.shape[1]
    readerE = pd.read_csv("/Directory/Energy4.csv", chunksize=n)
    for chunkE in readerE:
        nTS = 0
        if nE<nc:
            nE = nE+1
            continue
        elif nE>nc:
            break
        elif nE==nc:
            readerTS = pd.read_csv("TSPOS4.csv", chunksize=n)
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
                            for i in range(454):
                                if i == 0:
                                    Y1 = int(chunkTS['{}'.format('y1')][j])
                                    Y2 = int(chunkTS['{}'.format('y2')][j])
                                    Y3 = int(chunkTS['{}'.format('y3')][j])
                                    Y4 = int(chunkTS['{}'.format('y4')][j])
                                                                        
                                    X1 = int(chunkTS['{}'.format('x1')][j])
                                    X2 = int(chunkTS['{}'.format('x2')][j])
                                    X3 = int(chunkTS['{}'.format('x3')][j])
                                    X4 = int(chunkTS['{}'.format('x4')][j])
                                    if Y1 == Y3 and Y1==Y2 and Y1 == Y4 and X1 == X3 and X1==X2 and X1 == X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y2 and Y3==Y4 and X1 == X2 and X3==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                            
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y3 and Y2==Y4 and X1 == X3 and X2==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                             
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y4 and Y2==Y3 and X1 == X4 and X2==X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y2 and Y1==Y3 and X1 == X2 and X1==X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y2 and Y1==Y4 and X1 == X2 and X1==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y3 and Y1==Y4 and X1 == X3 and X1==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y3 and Y2==Y1 and X2 == X3 and X2==X1:
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y1 and Y2==Y4 and X2 == X1 and X2==X4:
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y3 and Y2==Y4 and X2 == X3 and X2==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                    elif Y3 == Y1 and Y2==Y3 and X3 == X1 and X2==X3:
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y3 and Y4==Y3 and X1 == X3 and X4==X3:
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y3 == Y2 and Y4==Y3 and X3 == X2 and X4==X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y4 == Y1 and Y4==Y2 and X4 == X1 and X4==X2:
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y4 == Y2 and Y4==Y3 and X4 == X2 and X4==X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y2 and X1 == X2:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y3 and X1 == X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y4 and X1 == X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y3 and X2 == X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y4 and X2 == X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y3 == Y4 and X3 == X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    else:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                            
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                            
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                                                            
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                elif i==1 or i==2 or i==3:
                                    continue
                                else:
                                  if pd.isna(chunkC['{}'.format(i-4)][j]) == False and pd.isna(chunkE['{}'.format(i-4)][j])==False:
                                        D = int(chunkC['{}'.format(i-4)][j])
                                        C.append(EcalID(D).cell())
                                        M.append(EcalID(D).module())
                                        L.append(EcalID(D).layer() + 1)
                                        E.append(chunkE['{}'.format(i-4)][j])
                            MNN = nx.Graph()
                            MNN.add_nodes_from([(k,{"pos":(float(C[k]),float(M[k]),float(L[k])),"x":[float(E[k]),float(C[k]),float(M[k]),float(L[k])]   }) for k in range(len(C))] )
                            Edges = nx.geometric_edges(MNN, radius=40)
                            MNN.add_edges_from(Edges)
                            MNN = from_networkx(MNN)
                            MNN["y"] = 3
                            torch.save(MNN,'/Directory/GNN/ETSX/4e{}.pt'.format(nc))
                            with zipfile.ZipFile('/Directory/GNN/ETSX/4e{}.zip'.format(nc), 'w', zipfile.ZIP_DEFLATED) as f:
                                    f.write('/Directory/GNN/ETSX/4e{}.pt'.format(nc))
                            os.remove('/Directory/GNN/ETSX/4e{}.pt'.format(nc))
                    elif NRR <n:
                        for j in range(nc*n,nc*n+NRR):
                            C = []
                            M = []
                            L = []
                            E = []
                            for i in range(454):
                                if i == 0:
                                    Y1 = int(chunkTS['{}'.format('y1')][j])
                                    Y2 = int(chunkTS['{}'.format('y2')][j])
                                    Y3 = int(chunkTS['{}'.format('y3')][j])
                                    Y4 = int(chunkTS['{}'.format('y4')][j])
                                                                        
                                    X1 = int(chunkTS['{}'.format('x1')][j])
                                    X2 = int(chunkTS['{}'.format('x2')][j])
                                    X3 = int(chunkTS['{}'.format('x3')][j])
                                    X4 = int(chunkTS['{}'.format('x4')][j])
                                    if Y1 == Y3 and Y1==Y2 and Y1 == Y4 and X1 == X3 and X1==X2 and X1 == X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y2 and Y3==Y4 and X1 == X2 and X3==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                            
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y3 and Y2==Y4 and X1 == X3 and X2==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                             
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y4 and Y2==Y3 and X1 == X4 and X2==X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y2 and Y1==Y3 and X1 == X2 and X1==X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y2 and Y1==Y4 and X1 == X2 and X1==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y3 and Y1==Y4 and X1 == X3 and X1==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y3 and Y2==Y1 and X2 == X3 and X2==X1:
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y1 and Y2==Y4 and X2 == X1 and X2==X4:
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y3 and Y2==Y4 and X2 == X3 and X2==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                    elif Y3 == Y1 and Y2==Y3 and X3 == X1 and X2==X3:
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y3 and Y4==Y3 and X1 == X3 and X4==X3:
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y3 == Y2 and Y4==Y3 and X3 == X2 and X4==X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y4 == Y1 and Y4==Y2 and X4 == X1 and X4==X2:
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y4 == Y2 and Y4==Y3 and X4 == X2 and X4==X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y2 and X1 == X2:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y3 and X1 == X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y4 and X1 == X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y3 and X2 == X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y4 and X2 == X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y3 == Y4 and X3 == X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    else:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                            
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                            
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                                                            
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                elif i==1 or i==2 or i==3:
                                    continue
                                else:
                                  if pd.isna(chunkC['{}'.format(i-4)][j]) == False and pd.isna(chunkE['{}'.format(i-4)][j])==False:
                                        D = int(chunkC['{}'.format(i-4)][j])
                                        C.append(EcalID(D).cell())
                                        M.append(EcalID(D).module())
                                        L.append(EcalID(D).layer() + 1)
                                        E.append(chunkE['{}'.format(i-4)][j])
                            MNN = nx.Graph()
                            MNN.add_nodes_from([(k,{"pos":(float(C[k]),float(M[k]),float(L[k])),"x":[float(E[k]),float(C[k]),float(M[k]),float(L[k])]   }) for k in range(len(C))] )
                            Edges = nx.geometric_edges(MNN, radius=40)
                            MNN.add_edges_from(Edges)
                            MNN = from_networkx(MNN)
                            MNN["y"] = 3
                            torch.save(MNN,'/Directory/GNN/ETSX/4e{}.pt'.format(nc))
                            with zipfile.ZipFile('/Directory/GNN/ETSX/4e{}.zip'.format(nc), 'w', zipfile.ZIP_DEFLATED) as f:
                                    f.write('/Directory/GNN/ETSX/4e{}.pt'.format(nc))
                            os.remove('/Directory/GNN/ETSX/4e{}.pt'.format(nc))
                    else:
                        for j in range(nc*NRR,(nc+1)*NRR):
                            C = []
                            M = []
                            L = []
                            E = []
                            for i in range(454):
                                if i == 0:
                                    Y1 = int(chunkTS['{}'.format('y1')][j])
                                    Y2 = int(chunkTS['{}'.format('y2')][j])
                                    Y3 = int(chunkTS['{}'.format('y3')][j])
                                    Y4 = int(chunkTS['{}'.format('y4')][j])
                                                                        
                                    X1 = int(chunkTS['{}'.format('x1')][j])
                                    X2 = int(chunkTS['{}'.format('x2')][j])
                                    X3 = int(chunkTS['{}'.format('x3')][j])
                                    X4 = int(chunkTS['{}'.format('x4')][j])
                                    if Y1 == Y3 and Y1==Y2 and Y1 == Y4 and X1 == X3 and X1==X2 and X1 == X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y2 and Y3==Y4 and X1 == X2 and X3==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                            
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y3 and Y2==Y4 and X1 == X3 and X2==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                             
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y4 and Y2==Y3 and X1 == X4 and X2==X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y2 and Y1==Y3 and X1 == X2 and X1==X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y2 and Y1==Y4 and X1 == X2 and X1==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y3 and Y1==Y4 and X1 == X3 and X1==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y3 and Y2==Y1 and X2 == X3 and X2==X1:
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y1 and Y2==Y4 and X2 == X1 and X2==X4:
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y3 and Y2==Y4 and X2 == X3 and X2==X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                    elif Y3 == Y1 and Y2==Y3 and X3 == X1 and X2==X3:
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y3 and Y4==Y3 and X1 == X3 and X4==X3:
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y3 == Y2 and Y4==Y3 and X3 == X2 and X4==X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y4 == Y1 and Y4==Y2 and X4 == X1 and X4==X2:
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y4 == Y2 and Y4==Y3 and X4 == X2 and X4==X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y2 and X1 == X2:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y3 and X1 == X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y1 == Y4 and X1 == X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y3 and X2 == X3:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                    elif Y2 == Y4 and X2 == X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    elif Y3 == Y4 and X3 == X4:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                              
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                    else:
                                            C.append(Y1)
                                            M.append(X1)
                                            L.append(0)
                                            E.append(1)
                                                                            
                                            C.append(Y2)
                                            M.append(X2)
                                            L.append(0)
                                            E.append(1)
                                                                            
                                            C.append(Y3)
                                            M.append(X3)
                                            L.append(0)
                                            E.append(1)
                                                                            
                                            C.append(Y4)
                                            M.append(X4)
                                            L.append(0)
                                            E.append(1)
                                elif i==1 or i==2 or i==3:
                                    continue
                                else:
                                  if pd.isna(chunkC['{}'.format(i-4)][j]) == False and pd.isna(chunkE['{}'.format(i-4)][j])==False:
                                        D = int(chunkC['{}'.format(i-4)][j])
                                        C.append(EcalID(D).cell())
                                        M.append(EcalID(D).module())
                                        L.append(EcalID(D).layer() + 1)
                                        E.append(chunkE['{}'.format(i-4)][j])
                            MNN = nx.Graph()
                            MNN.add_nodes_from([(k,{"pos":(float(C[k]),float(M[k]),float(L[k])),"x":[float(E[k]),float(C[k]),float(M[k]),float(L[k])]   }) for k in range(len(C))] )
                            Edges = nx.geometric_edges(MNN, radius=40)
                            MNN.add_edges_from(Edges)
                            MNN = from_networkx(MNN)
                            MNN["y"] = 3
                            torch.save(MNN,'/Directory/GNN/ETSX/4e{}.pt'.format(nc))
                            with zipfile.ZipFile('/Directory/GNN/ETSX/4e{}.zip'.format(nc), 'w', zipfile.ZIP_DEFLATED) as f:
                                    f.write('/Directory/GNN/ETSX/4e{}.pt'.format(nc))
                            os.remove('/Directory/GNN/ETSX/4e{}.pt'.format(nc))
                    nTS = nTS+1
            nE = nE+1
    nc = nc+1
