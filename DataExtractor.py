#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import uproot as up

class DataExtractor():#This class is for extracting different parts of the data from the different ROOT files using Uproot 
    def __init__(self,Inputname,OutputNumber,Length, output_path = ""):
        self.Inputname  = Inputname #This is the name of the data file containing the data 
        self.OutputNumber  = OutputNumber #This number will be used to name our output file so for example EcalID1 if you input 1
        self.Length = Length#This is the length of an array used to save the data so that it becomes uniform, for 1e the data used for this project had a maximum length of 155, 2e: 250, 3e:360, and 4e:450. 
        self.output_path = output_path 
    def Interloper(self):#Interloper is responsible for extracting the Energy and the EcalID of each ROOT file that contains 10000 events each. If the (X,Y,Z) coordinates want to be accessed then the file can be amended to save those as well
        l = 0  #Simply counts the iterations
        for i in range(len(self.Inputname)):#This iterates through the different file names 
            EcalID = []#Each value is initially saved into a smaller list containing other hits from the event and are all appended to this list
            Energy = []
            tree = up.open(self.Inputname[i])#Opens up the ROOT file 
            KEYS = tree.keys()#This extracts the name of the events in the ROOT file, if everything has gone well during data production the length of this should be non-zero
            if len(KEYS)>0:#This means that we only try and load data from files that actually have keys as otherwise it cannot be loaded.
                branches = tree[KEYS[1]]["EcalRecHits_sim"].arrays(library = "np")#This loads all the data from the simulated Ecal 
                IDS = branches["EcalRecHits_sim.id_"]#This loads the EcalID data from all the events in the ROOT file 
                ES = branches["EcalRecHits_sim.energy_"]#This loads the energy data from all the events in the ROOT file 
                for j in range(len(IDS)):#This will iterate through the length of total events in the EcalID file 
                    EID = [np.nan] * self.Length#This is a temporary list for the EcalID that we make sure is padded to make the events have a certain length when saved, for 1e the data used for this project had a maximum length of 155, 2e: 250, 3e:360, and 4e:450. 
                    E = [np.nan] * self.Length#Same as for the EcalID but for the energy 
                    for k in range(len(IDS[j])):#Iterates through the hits in each event to extract the values 
                        D = float(IDS[j][k])#Loads the EcalID value for each hit 
                        DE = float(ES[j][k])#Loads the Energy value for each hit 
                        EID[k] =  round(D,2)#Exchanges each hit value in the array from NAN to the EcalID
                        E[k] =  round(DE,2)#Exchanges each hit value in the array from NAN to the EcalID
                    EcalID.append(EID)#Appends the list of hits in the specific event to the list of lists 
                    Energy.append(E)
                EDF = pd.DataFrame(Energy)#To save the file we used Pandas so we convert the list of lists into a data frame that we can then easily convert into a csv file 
                IDF = pd.DataFrame(EcalID)
                print("DataFrame: Done")
                if l ==0:#This creates the files on the first iteration and the rest of the data is then appended to the end of this file                                                         
                    IDF.to_csv(self.output_path + "EcalID{}.csv".format(self.OutputNumber),mode='w',chunksize=5000,index=False)#This converts the data frame object into a csv file, the chunksize is arbitrary but was found to be faster on my machine than just writing everything in 10000 chunks (Change as you see fit)
                    EDF.to_csv(self.output_path + "Energy{}.csv".format(self.OutputNumber),mode='w',chunksize=5000,index=False)
                    print("CSV: Done")
                else:
                    IDF.to_csv(self.output_path + "EcalID{}.csv".format(self.OutputNumber),mode='a',chunksize=5000,index=False, header=False)#This simply appends the data to the already created file (mode='a')
                    EDF.to_csv(self.output_path + "Energy{}.csv".format(self.OutputNumber),mode='a',chunksize=5000,index=False, header=False)
                    print("CSV: Done")
                l = l+1
    def BREM(self):#This is responsible for extracting which events are Brem events which is useful for later analysis 
        PhotonNumber = []#This will contain how many photons are in each event 
        EventNumber = []#This contains which event number has that number of photons 
        for i in range(len(self.Inputname)):#Loops over all ROOT files 
            tree = up.open(self.Inputname[i])
            KEYS = tree.keys()
            print(KEYS)
            if len(KEYS) >0:
                B3 = tree[KEYS[1]]["EcalScoringPlaneHits_sim"].arrays(library = "np")#This loads data from the scoring plane of the simulated Ecal 
                EScoring  = B3["EcalScoringPlaneHits_sim.energy_"]#This loads the energy of the incident particles that can now be electrons, photons, or other particles produced 
                PDGScoring = B3["EcalScoringPlaneHits_sim.pdgID_"]#The pdgID helps identify what type of particle has hit the scoring plane 
                B4 = tree[KEYS[1]]["EcalVeto_sim"].arrays(library = "np")#We load this data from the EcalVeto to determine if there are photons in the event to begin with 
                PH = B4["photonContainmentEnergy_"]#We use the photon Containment Energy to determine if there are photons passing through the Ecal
                for j in range(len(EScoring)):#This goes through the data points in the scoring plane 
                    if np.sum(PH[j]) !=0:#We sum over the photon containment energy and if it is non-zero that indicates that photons were produced in the event and so we start to try and determine how many 
                        EE =  EScoring[i]#The energy 
                        PDG = PDGScoring[i]#The PDG ID 
                        P = 0#This is used to count how many photons are present 
                        for k in range(len(EE)):#This loops through the scoring plane particles 
                            if PDG[k] == 22 and EE[k]>10:#If the PDG ID is for a photon and the energy is above 10 MeV (A cutoff due to sensitivity of the detector) then that is counted as a photon 
                                P = P+1
                        if P>0:#If the number of photons P is larger than zero then we actually save that event 
                            PhotonNumber.append(P)
                            EventNumber.append(10000*i+j)
        Data = {"EN":EventNumber, "PN": PhotonNumber}#Our data will have two columns: EN for event number and PN for photon number 
        DDF = pd.DataFrame(Data)#This converts our data into a dataframe 
        DDF.to_csv(self.output_path + "Brem{}.csv".format(self.OutputNumber))#Converts to a csv file 
    def TSPOS(self):
        Positions = []
        for i in range(len(self.Inputname)):#Loopd over the files 
            tree = up.open(self.Inputname[i])
            KEYS = tree.keys()
            if len(KEYS)>0:
                branches = tree[KEYS[1]]["BeamElectronTruthInfo_sim"].arrays(library = "np")#We load the TrigScintScoringPlaneHits_sim from the file 
                XPOS = branches["BeamElectronTruthInfo_sim.barX_"]#The x-coordinate data 
                YPOS = branches["BeamElectronTruthInfo_sim.barY_"]#The y-coordinate data
                for j in range(len(XPOS)):
                    POS = np.zeros(len(XPOS[j])*2, dtype=int)
                    for i in range(len(XPOS[j])):
                        POS[i*2] = XPOS[j][i]
                        POS[i*2 + 1] = YPOS[j][i]
                    Positions.append(POS)
            EDF = pd.DataFrame(Positions)
            total_columns = int(len(EDF.columns)/2)
        DCF = pd.DataFrame(columns = [(x + "{}".format(i)) for i in range(1, total_columns+1) for x in ("X", "Y")])#creates a header for the csv based on the OutputNumber parameter
        DCF.to_csv(self.output_path + "POS{}.csv".format(self.OutputNumber), mode="w",index=False)
        EDF.to_csv(self.output_path + "POS{}.csv".format(self.OutputNumber),mode='a',index=False,header=False)
        print("CSV: Done")

    





