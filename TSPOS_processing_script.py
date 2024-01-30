from TriggerScintillator import TS1, TS2, TS3, TS4

output_directory = "Output/TSPOS/"
POS_files = ['Output/POS/POS1.csv','Output/POS/POS2.csv','Output/POS/POS3.csv', 'Output/POS/POS4.csv']

TS1.Process_POS_1_event(POS_files[0], output_directory)
print("Finished TS1")
TS2.Process_POS_2_events(POS_files[1], output_directory)
print("Finished TS2")
TS3.Process_POS_3_events(POS_files[2], output_directory)
print("Finished TS3")
TS4.Process_POS_4_events(POS_files[3], output_directory)
print("Finished TS3")