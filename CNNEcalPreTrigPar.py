from CNN_and_Combined_PreProcessing import CNNEcalPooledTrigger as Ecal
import multiprocessing as mp

ecal_path = "/data/LDMXML/8mil_processing/EnergyEcalPosData/"
energy_path = "/data/LDMXML/8mil_processing/EnergyEcalPosData/"
output_path = "/data/LDMXML/8mil_processing/CNNEcalTrigPooledFast/"
print("Started pooling trigger")
p1 = mp.Process(target=Ecal.CNN_Ecal, args=[1,ecal_path + "EcalID1.csv", energy_path + "Energy1.csv", output_path,1000, 155])
print("Finished Ecal 1 event")
p2 = mp.Process(target=Ecal.CNN_Ecal,args=[2,ecal_path + "EcalID2.csv", energy_path + "Energy2.csv", output_path,1000,250])
print("Finished Ecal 2 event")
p3 = mp.Process(target=Ecal.CNN_Ecal,args=[3,ecal_path + "EcalID3.csv", energy_path + "Energy3.csv", output_path,1000,400])
print("Finished Ecal 3 events")
p4 = mp.Process(target = Ecal.CNN_Ecal, args=[4,ecal_path + "EcalID4.csv", energy_path + "Energy4.csv", output_path, 1000, 450])
print("Finished Ecal 4 events")
p1.start()
p2.start()
p3.start()
p4.start()

