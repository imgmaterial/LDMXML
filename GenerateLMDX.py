#!/bin/python

import libDetDescr

#import uproot as up

import sys

import matplotlib.pyplot as plt

import numpy as np


import importlib.util
import sys

import pandas as pd

from libDetDescr import EcalID, HcalID

import csv

import uproot as up

tree = up.open("simoutput.root")

treeevent = tree["LDMX_Events;5"]

branches  = treeevent["EcalRecHits_sim"].arrays(library = "np")



class DataExtract():
    def __init__(self,Inputname,Event,OutputNumber):
        self.Inputname  = Inputname
        self.Event = Event
        self.OutputNumber  = OutputNumber
    def Producer(self):
        #Combine Energy into a CSV file as well, go around having to extract it later
        tree = up.open(self.Inputname)
        Cell = []
        Module = []
        Layers = []
        Energy = []
        
        for i in range(len(self.Event)):
            branches = tree[self.Event[i]]["EcalRecHits_sim"].arrays(library = "np")
            IDS = branches["EcalRecHits_sim.id_"]
            #print(len(IDS))
            for j in range(len(IDS)):
                C = []
                M = []
                L = []
                E = []
                for k in range(len(IDS[j])):
                    D = int(IDS[j][k])
                    C.append(EcalID(D).cell())
                    M.append(int(EcalID(D).module()))
                    L.append(int(EcalID(D).layer()))
                    E.append(branches["EcalRecHits_sim.energy_"][j][k])
                Cell.append(C)
                Module.append(M)
                Layers.append(L)
                Energy.append(E)
        CDF = pd.DataFrame(Cell)#,columns = ["Cell","Module","Layer"])
        MDF = pd.DataFrame(Module)
        LDF = pd.DataFrame(Layers)
        EDF = pd.DataFrame(Energy)
        
        CDF.to_csv("Cell{}.csv".format(self.OutputNumber), index=False)
        MDF.to_csv("Module{}.csv".format(self.OutputNumber), index=False)
        LDF.to_csv("Layer{}.csv".format(self.OutputNumber), index=False)
        EDF.to_csv("Energy{}.csv".format(self.OutputNumber), index=False)
        
        
        #return len(Cell),len(Module),len(Layers),len(Energy)




class PCSV():
    def __init__(self,Input,Output):
        self.Input  = Input
        self.Output  = Output
    def P(self):
        
        DF = pd.read_csv(self.Input)

        df = DF.fillna(0)

        NR = len(df.axes[0])

        NC = len(df.axes[1])
        
        Cell = []

        Module = []

        Layer = []

        for i in range(NR):
            Ce = []
            Mo =  []
            La = []
            for j in range(NC):
                D = int(df['{}'.format(j)][i])
                #print(D)
                #print()
                if D != 0:
                    Ce.append(int(EcalID(D).cell()))
                    Mo.append(int(EcalID(D).module()))
                    La.append(int(EcalID(D).layer()))
            Cell.append(Ce)
            Module.append(Mo)
            Layer.append(La)
        #print(Module[0][0])
        
        CellIDS = [Cell,Module,Layer]
        
        #with open("Processed2Ex.csv", "w", newline="") as f:
        #    writer = csv.writer(f)
        #    writer.writerow(Cell)
        #    writer.writerow(Module)
        #    writer.writerow(Layer)
            #writer.writerows(CellIDS)
        
        C = pd.DataFrame(Cell)#,columns = ["Cell","Module","Layer"])
        M = pd.DataFrame(Module)
        L = pd.DataFrame(Layer)
        
        C.to_csv("Cell{}.csv".format(self.Output), index=False)
        M.to_csv("Module{}.csv".format(self.Output), index=False)
        L.to_csv("Layer{}.csv".format(self.Output), index=False)
        #print(L)


#DataExtract("simoutput.root",["LDMX_Events;6"],1).Producer()

#DataExtract("simoutput2.root",["LDMX_Events;11"],2).Producer()

DataExtract("testoutput1.root",["LDMX_Events;2"],'1t').Producer()

DataExtract("testoutput2.root",["LDMX_Events;3"],'2t').Producer()


DF = pd.read_csv("Cell1t.csv")

print(DF)

#DF2 = pd.read_csv("Cell2.csv")

#print(DF2)


#,"LDMX_Events;6"


#PCSV("CellIDTwo.csv",2).P()

#PCSV("CellIDOne.csv",1).P()

#PCSV("CellIDTwoT.csv","2t").P()

#PCSV("CellIDOneT.csv","1t").P()

#DN = pd.read_csv('CellIDTwo.csv')


#DF = pd.read_csv('Layer1.csv')

#print(DN)

#print(DF)






#df = DF.fillna(0)

#NR = len(df.axes[0])

#NC = len(df.axes[1])
#print(int(df['{}'.format(115)][0]))

#import awkward as ak






#D = []

#Cell = []

#Module = []

#Layer = []

#print(int(0.0))

#for i in range(NR):
#        Ce = []
#        Mo = []
#        La = []
#        for j in range(NC):
#            D = int(df['{}'.format(j)][i])
            #print(D)
            #print()
#            if D != 0:
#                Ce.append(EcalID(D).cell() )
#                Mo.append(EcalID(D).module() )
#                La.append(EcalID(D).layer())
#        Cell.append(Ce)
#        Module.append(Mo)
#        Layer.append(La)

#print(Cell)


#CML = pd.DataFrame([Cell,Module,Layer])#,columns = ["Cell","Module","Layer"])

#CML.to_csv('Processed1Ex.csv', index=False)

#T = EcalID(D[0][0])

#for i in range(10):
    #print(np.max(Cell[i]))
#print(Cell[0][120])
#print(Module[0][120])
#print(Layer[0][120])

#print(T.module())

#T = D[0][0]

#tree = up.open("simoutput.root")

#treeevent = tree["LDMX_Events;1"]

#branches  = treeevent["EcalRecHits_sim"].arrays()

#from_raw = EcalID(335810563)
#one = EcalID(335810563)
#print("EcalID [raw, cell, module, layer]  \
#({from_raw.raw()}, {from_raw.cell()}, {from_raw.module()}, {from_raw.layer()})")
#print(from_raw.cell())
#print(one.module(),one.cell(),one.layer())
