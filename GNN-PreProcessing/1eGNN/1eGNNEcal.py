import torch #The Graph generation relies on pytorch for the general structure and saving of the graph.

from torch_geometric.utils import from_networkx#The graph is built with networkx as it provides simple tools for building graphs but is converted from networkx to pytorch

import pandas as pd

import numpy as np

import networkx as nx #When using jupyter there might be problems when loading the modules, make sure to import torch and then import torch_geometric.utils

import os#This is needed for how the files are saved

from zipfile import ZipFile#To ensure that the data is saved in a compress format I ended using Zip files, other methods exist but have not been attempted yet

import zipfile

n = 1000

nc = 0

nE = 0

nTS = 0

import libDetDescr

from libDetDescr import EcalID, HcalID,EcalTriggerID


readerC = pd.read_csv("/Directory/EcalIDFile1.csv", chunksize=n)#Chunks the EcalID file


for chunkC in readerC:
    nE = 0
    NRR = chunkC.shape[0]
    NC = chunkC.shape[1]
    D = []
    readerE = pd.read_csv("/Directory/Energy1.csv", chunksize=n)
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
                    C = []#It is easier to create the graph if it can internally loop over a bunch of  lists so we extract the values here before continuing
                    M = []
                    L = []
                    E = []
                    for i in range(155):
                            if pd.isna(chunkC['{}'.format(i)][j]) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                            D = int(chunkC['{}'.format(i)][j])
                            C.append(EcalID(D).cell())
                            M.append(EcalID(D).module())
                            L.append(EcalID(D).layer())
                            E.append(chunkE['{}'.format(i)][j])#Appends the data for the different lists
                    MNN = nx.Graph()#This creates a graph object using networkx, MNN is then the graph object that you add or remove nodes from
                    MNN.add_nodes_from([(k,{"pos":(float(C[k]),float(M[k]),float(L[k])),"x":[float(E[k]),float(C[k]),float(M[k]),float(L[k])]   }) for k in range(len(C))] )#This adds the nodes where each data point is a node in the graph. The position of each node is (C,M,L) and is passed to the graph under the 'pos' label. The feature of each node is [Energy, Cell, Module, Layer] and is passed under 'x' label.
                    Edges = nx.geometric_edges(MNN, radius=40)#The edges are what connect the different nodes and here undirected edges are used. The actual edges are calculated using the geometric radius of a sphere around each node with the radius set to 40, so all nodes within the radius of a node are then connected. Exploring what types of edges are best still needs to be done.
                    MNN.add_edges_from(Edges)#This simply adds the calculated edges to the graph object
                    MNN = from_networkx(MNN)#This converts the graph from a networkx object to a pytorch graph that can be saved
                    MNN["y"] = 0#This assigns a category to each graph
                    torch.save(MNN,'/Directory/GNN/Ecal/1e{}.pt'.format(nc))#This saves the graph to a .pt file format that can be loaded with pytorch
                    with zipfile.ZipFile('/Directory/GNN/Ecal/1e{}.zip'.format(nc), 'w', zipfile.ZIP_DEFLATED) as f:#This zips the file
                            f.write('/Directory/GNN/Ecal/1e{}.pt'.format(nc))
                    os.remove('/Directory/GNN/Ecal/1e{}.pt'.format(nc))#This removes the original file only keeping the .zip file instead 
            elif NRR <n:
                for j in range(nc*n,nc*n+NRR):
                    C = []
                    M = []
                    L = []
                    E = []
                    for i in range(155):
                            if pd.isna(chunkC['{}'.format(i)][j]) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                            D = int(chunkC['{}'.format(i)][j])
                            C.append(EcalID(D).cell())
                            M.append(EcalID(D).module())
                            L.append(EcalID(D).layer())
                            E.append(chunkE['{}'.format(i)][j])
                    MNN = nx.Graph()
                    MNN.add_nodes_from([(k,{"pos":(float(C[k]),float(M[k]),float(L[k])),"x":[float(E[k]),float(C[k]),float(M[k]),float(L[k])]   }) for k in range(len(C))] )
                    Edges = nx.geometric_edges(MNN, radius=40)
                    MNN.add_edges_from(Edges)
                    MNN = from_networkx(MNN)
                    MNN["y"] = 0
                    torch.save(MNN,'/Directory/GNN/Ecal/1e{}.pt'.format(nc))
                    with zipfile.ZipFile('/Directory/GNN/Ecal/1e{}.zip'.format(nc), 'w', zipfile.ZIP_DEFLATED) as f:
                            f.write('/Directory/GNN/Ecal/1e{}.pt'.format(nc))
                    os.remove('/Directory/GNN/Ecal/1e{}.pt'.format(nc))
            else:
                for j in range(nc*NRR,(nc+1)*NRR):
                    C = []
                    M = []
                    L = []
                    E = []
                    for i in range(155):
                        if pd.isna(chunkC['{}'.format(i)][j]) == False and pd.isna(chunkE['{}'.format(i)][j])==False:
                            D = int(chunkC['{}'.format(i)][j])
                            C.append(EcalID(D).cell())
                            M.append(EcalID(D).module())
                            L.append(EcalID(D).layer())
                            E.append(chunkE['{}'.format(i)][j])
                    MNN = nx.Graph()
                    MNN.add_nodes_from([(k,{"pos":(float(C[k]),float(M[k]),float(L[k])),"x":[float(E[k]),float(C[k]),float(M[k]),float(L[k])]   }) for k in range(len(C))] )
                    Edges = nx.geometric_edges(MNN, radius=40)
                    MNN.add_edges_from(Edges)
                    MNN = from_networkx(MNN)
                    MNN["y"] = 0
                    torch.save(MNN,'/Directory/GNN/Ecal/1e{}.pt'.format(nc))
                    with zipfile.ZipFile('/Directory/GNN/Ecal/1e{}.zip'.format(nc), 'w', zipfile.ZIP_DEFLATED) as f:
                            f.write('/Directory/GNN/Ecal/1e{}.pt'.format(nc))
                    os.remove('/Directory/GNN/Ecal/1e{}.pt'.format(nc))
            nE = nE+1
    nc = nc+1
