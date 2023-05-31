#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd 

import numpy as np

import sys

import matplotlib.pyplot as plt

import numpy as np


import importlib.util
import sys

import pandas as pd

#from libDetDescr import EcalID, HcalID

import csv

import uproot as up




DF = pd.read_csv('Directory/LIST2.csv') #LIST2 is a list of file names that contains the root files with the data, each one contained 10000 data points for this specific project 

DF.columns = ['0']

NR = len(DF.axes[0])

CLIST = []




class DataExtractor():
    def __init__(self,Inputname,Event,OutputNumber,N):
       self.Inputname  = Inputname
       self.Event = Event
       self.OutputNumber  = OutputNumber
       self.N = N
    def Interloper(self):
        EcalID = []
        
       	Brem = []
        
        Energy = []
       	l = -1  
        for i in range(len(self.Inputname)):
            EcalID = []
            M=0
            #Brem = []
            Energy = []
            #X = []
            #Y = []
            #Z = []
            
            tree = up.open(self.Inputname[i])
            
            KEYS = tree.keys()
            print(KEYS)
            
            if len(KEYS)>0:
                
                 l = l+1
                 branches = tree[KEYS[0]]["EcalRecHits_sim"].arrays(library = "np")
                 
                 IDS = branches["EcalRecHits_sim.id_"]
                 
                 ES = branches["EcalRecHits_sim.energy_"]
                
                 
                 for j in range(len(IDS)):
                     EID = [np.nan] * 250
                     
                     E = [np.nan] * 250
                     
                     #XM = []
                     
                     #YM = []
                     
                     #ZM = []
                     
                     for k in range(len(IDS[j])):
                           D = float(IDS[j][k])
                           DE = float(ES[j][k])
                           EID[k]=  round(D,2)
                           E[k]=  round(D,2)
                     #      XM.append(XS[j][k])
                     #      YM.append(YS[j][k])
#                           ZM.append(XS[j][k])
                     EcalID.append(EID)
                     Energy.append(EID)
                     #X.append(XM)
                     #Y.append(YM)
                     #Z.append(ZM)
                 #print(len(Energy))
                 EDF = pd.DataFrame(Energy)
                 IDF = pd.DataFrame(EcalID)
                 #print("DataFrame: Done")
                     
                 if l ==0:
                          IDF.to_csv("/home/jacoblindahl/EcalID{}.csv".format(self.OutputNumber),mode='w',chunksize=5000,index=False)
                          EDF.to_csv("/home/jacoblindahl/Energy{}.csv".format(self.OutputNumber),mode='w',chunksize=5000,index=False)
                          print("CSV: Done")
                          print(0)
                 else:
                          IDF.to_csv("/home/jacoblindahl/EcalID{}.csv".format(self.OutputNumber),mode='a',chunksize=5000,index=False, header=False)
                          EDF.to_csv("/home/jacoblindahl/Energy{}.csv".format(self.OutputNumber),mode='a',chunksize=5000,index=False, header=False)
                          print("CSV: Done",1)
                 print(i)
    def BREM(self):
        N=0
        PhotonNumber = []
        EventNumber = []
        for i in range(len(self.Inputname)):
            tree = up.open(self.Inputname[i])
            
            KEYS = tree.keys()
            print(KEYS)
            if len(KEYS) >0:
                  N = N+1
                   
                  B3 = tree[KEYS[0]]["EcalScoringPlaneHits_sim"].arrays(library = "np")
                  EScoring  = B3["EcalScoringPlaneHits_sim.energy_"]

                  PDGScoring = B3["EcalScoringPlaneHits_sim.pdgID_"]
                  B4 = tree[KEYS[0]]["EcalVeto_sim"].arrays(library = "np")
                  PH = B4["photonContainmentEnergy_"]
                  for j in range(len(EScoring)):
                           if np.sum(PH[j]) !=0:
                                      EE =  EScoring[i]
                                      PDG = PDGScoring[i]
                                      P = 0
                                      for k in range(1,len(EE)):
                                            if PDG[k] == 22 and EE[k]>10:
                                                      P = P+1
                                      if P>0:
                                             PhotonNumber.append(P)
                                             EventNumber.append(i*1000+j)

        Data = {"EN":EventNumber, "PN": PhotonNumber}
        DDF = pd.DataFrame(Data)
        DDF.to_csv("Brem{}.csv".format(self.OutputNumber))
    def  Trig(self):
         x = []
         y = []
         
         B = 4
         l = -1
    
         for i in range(len(self.Inputname)):
            Positions = []
            #y = []
            
            tree = up.open(self.Inputname[i])
            
            KEYS = tree.keys()
            
            print(KEYS)
            
            if len(KEYS)>0:
                 l = l+1
                 branches = tree[KEYS[0]]["TrigScintScoringPlaneHits_sim"].arrays(library = "np")
                 XPOS = branches["TrigScintScoringPlaneHits_sim.x_"]
                 YPOS = branches["TrigScintScoringPlaneHits_sim.y_"]
                 if B ==1:
                         for j in range(len(XPOS)):
                               POS = []

                               POS.append(XPOS[j][0])
                               POS.append(YPOS[j][0])
                               Positions.append(POS)
                         print(len(Positions))
                         EDF = pd.DataFrame(Positions)
                         print("DataFrame: Done")
                     
                         if l ==0:
                                DCF = pd.DataFrame(columns = ['{}'.format(i) for i in range(2)])
                                DCF.to_csv("POS{}.csv".format(self.OutputNumber), mode="w",index=False)
                                EDF.to_csv("POS{}.csv".format(self.OutputNumber),mode='a',chunksize=5000,index=False,header=False)
                                print("CSV: Done")
                         else:
                                 EDF.to_csv("POS{}.csv".format(self.OutputNumber),mode='a',chunksize=5000,index=False, header=False)
                                 print("CSV: Done")
                         print(i)
                 elif B==2:
                         for j in range(len(XPOS)):
                               POS = []

                               POS.append(XPOS[j][0])
                               POS.append(YPOS[j][0])
                               POS.append(XPOS[j][1])
                               POS.append(YPOS[j][1])
                               Positions.append(POS)
                         print(len(Positions))
                         EDF = pd.DataFrame(Positions)
                         print("DataFrame: Done")
                     
                         if l ==0:
                                DCF = pd.DataFrame(columns = ['X1','Y1','X2','Y2'])
                                DCF.to_csv("POS{}.csv".format(self.OutputNumber), mode="w",index=False)
                                EDF.to_csv("POS{}.csv".format(self.OutputNumber),mode='a',chunksize=5000,index=False,header=False)
                                print("CSV: Done")
                         else:
                                 EDF.to_csv("POS{}.csv".format(self.OutputNumber),mode='a',chunksize=5000,index=False, header=False)
                                 print("CSV: Done")
                         print(i)

                 elif B==3:
                         for j in range(len(XPOS)):
                               POS = []

                               POS.append(XPOS[j][0])
                               POS.append(YPOS[j][0])
                               POS.append(XPOS[j][1])
                               POS.append(YPOS[j][1])
                               POS.append(XPOS[j][2])
                               POS.append(YPOS[j][2])
                               Positions.append(POS)
                         print(len(Positions))
                         EDF = pd.DataFrame(Positions)
                         print("DataFrame: Done")
                     
                         if l ==0:
                                DCF = pd.DataFrame(columns = ['X1','Y1','X2','Y2','X3','Y3'])
                                DCF.to_csv("POS{}.csv".format(self.OutputNumber), mode="w",index=False)
                                EDF.to_csv("POS{}.csv".format(self.OutputNumber),mode='a',chunksize=5000,index=False,header=False)
                                print("CSV: Done")
                         else:
                                 EDF.to_csv("POS{}.csv".format(self.OutputNumber),mode='a',chunksize=5000,index=False, header=False)
                                 print("CSV: Done")
                         print(i)
                 elif B==4:
                         for j in range(len(XPOS)):
                               POS = []

                               POS.append(XPOS[j][0])
                               POS.append(YPOS[j][0])
                               POS.append(XPOS[j][1])
                               POS.append(YPOS[j][1])
                               POS.append(XPOS[j][2])
                               POS.append(YPOS[j][2])
                               POS.append(XPOS[j][3])
                               POS.append(YPOS[j][3])
                               Positions.append(POS)
                         print(len(Positions))
                         EDF = pd.DataFrame(Positions)
                         print("DataFrame: Done")
                     
                         if l ==0:
                                DCF = pd.DataFrame(columns = ['X1','Y1','X2','Y2','X3','Y3','X4','Y4'])
                                DCF.to_csv("POS{}.csv".format(self.OutputNumber), mode="w",index=False)
                                EDF.to_csv("POS{}.csv".format(self.OutputNumber),mode='a',chunksize=5000,index=False,header=False)
                                print("CSV: Done")
                         else:
                                 EDF.to_csv("POS{}.csv".format(self.OutputNumber),mode='a',chunksize=5000,index=False, header=False)
                                 print("CSV: Done")
                         print(i)



DataExtractor(["DataDirectory/{}".format(DF['0'][i]) for i in range(NR)],"Remnant",1,10000).Interloper()










