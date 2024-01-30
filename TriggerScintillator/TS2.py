import pandas as pd

import numpy as np


#This file assumes that the columns in the .csv file containing the TS positional data are named X1, Y1, X2, and Y2

def Process_POS_2_events(input_file:str, output_directory:str):
    df = pd.read_csv(input_file)
    NR = len(df.axes[0])#The number of events in the file

    X1 = [df['X1'][i] for i in range(NR)]#Saves the X1 coordinates of the TS hits

    Y1 = [df['Y1'][i] for i in range(NR)]#Saves the Y1 coordinates of the TS hits


    X2 = [df['X2'][i] for i in range(NR)]#Saves the X2 coordinates of the TS hits

    Y2 = [df['Y2'][i] for i in range(NR)]#Saves the Y2 coordinates of the TS hits


    YT1 = [] #This is where the converted Y1 coordinate will be saved

    YT2 = [] #This is where the converted Y2 coordinate will be saved

    for BB in Y1:#This goes through the Y1 values and sorts them into 24 different categories
        if BB <= -33.0:
            YT1.append(0)
        elif -33.0<BB <= -30.0:
                YT1.append(1)
        elif -30.0<BB <= -27.0:
                YT1.append(2)
        elif -27.0<BB <= -24.0:
                YT1.append(3)
        elif -24.0<BB <= -21.0:
            YT1.append(4)
        elif -21.0<BB <= -18.0:
            YT1.append(5)
        elif -18.0<BB <= -15.0:
            YT1.append(6)
        elif -15.0<BB <= -12.0:
            YT1.append(7)
        elif -12.0<BB <= -9.0:
            YT1.append(8)
        elif -9.0<BB <= -6.0:
            YT1.append(9)
        elif -6.0<BB <= -3.0:
            YT1.append(10)
        elif -3.0<BB <= 0.0:
            YT1.append(11)
        elif 0.0<BB <= 3.0:
            YT1.append(12)
        elif 3.0<BB <= 6.0:
            YT1.append(13)
        elif 6.0<BB <= 9.0:
            YT1.append(14)
        elif 9.0<BB <= 12.0:
            YT1.append(15)
        elif 12.0<BB <= 15.0:
            YT1.append(16)
        elif 15.0<BB <= 18.0:
            YT1.append(17)
        elif 18.0<BB <= 21.0:
            YT1.append(18)
        elif 21.0<BB <= 24.0:
            YT1.append(19)
        elif 24.0<BB <= 27.0:
            YT1.append(20)
        elif 27.0<BB <= 30.0:
            YT1.append(21)
        elif 30.0<BB <= 33.0:
            YT1.append(22)
        elif BB >33.0:
            YT1.append(23)

    for BB in Y2:#This goes through the Y2 values and sorts them into 24 different categories
        if BB <= -33.0:
            YT2.append(0)
        elif -33.0<BB <= -30.0:
                YT2.append(1)
        elif -30.0<BB <= -27.0:
                YT2.append(2)
        elif -27.0<BB <= -24.0:
                YT2.append(3)
        elif -24.0<BB <= -21.0:
            YT2.append(4)
        elif -21.0<BB <= -18.0:
            YT2.append(5)
        elif -18.0<BB <= -15.0:
            YT2.append(6)
        elif -15.0<BB <= -12.0:
            YT2.append(7)
        elif -12.0<BB <= -9.0:
            YT2.append(8)
        elif -9.0<BB <= -6.0:
            YT2.append(9)
        elif -6.0<BB <= -3.0:
            YT2.append(10)
        elif -3.0<BB <= 0.0:
            YT2.append(11)
        elif 0.0<BB <= 3.0:
            YT2.append(12)
        elif 3.0<BB <= 6.0:
            YT2.append(13)
        elif 6.0<BB <= 9.0:
            YT2.append(14)
        elif 9.0<BB <= 12.0:
            YT2.append(15)
        elif 12.0<BB <= 15.0:
            YT2.append(16)
        elif 15.0<BB <= 18.0:
            YT2.append(17)
        elif 18.0<BB <= 21.0:
            YT2.append(18)
        elif 21.0<BB <= 24.0:
            YT2.append(19)
        elif 24.0<BB <= 27.0:
            YT2.append(20)
        elif 27.0<BB <= 30.0:
            YT2.append(21)
        elif 30.0<BB <= 33.0:
            YT2.append(22)
        elif BB >33.0:
            YT2.append(23)



    XT1 = []#Saves the X1 coordinate that has been sorted into 8 different categories
    XT2 = []#Saves the X2 coordinate that has been sorted into 8 different categories

    for BX in X1:
        if BX <= -9.0:
            XT1.append(0)
        elif -9.0 < BX <= -6.0:
            XT1.append(1)
        elif -6.0 < BX <= -3.0:
            XT1.append(2)
        elif -3.0 < BX <= 0:
            XT1.append(3)
        elif 0.0 < BX <= 3.0:
            XT1.append(4)
        elif 3.0 < BX <= 6.0:
            XT1.append(5)
        elif 6.0 < BX <= 9.0:
            XT1.append(6)
        elif BX > 9.0:
            XT1.append(7)

    for BX in X2:
        if BX <= -9.0:
            XT2.append(0)
        elif -9.0 < BX <= -6.0:
            XT2.append(1)
        elif -6.0 < BX <= -3.0:
            XT2.append(2)
        elif -3.0 < BX <= 0:
            XT2.append(3)
        elif 0.0 < BX <= 3.0:
            XT2.append(4)
        elif 3.0 < BX <= 6.0:
            XT2.append(5)
        elif 6.0 < BX <= 9.0:
            XT2.append(6)
        elif BX > 9.0:
            XT2.append(7)
    P = {"X1": XT1,"Y1":YT1,"X2":XT2, "Y2":YT2}#Creates a dictionary of the data

    PDF = pd.DataFrame(P)#Creates a Data Frame

    PDF.to_csv(output_directory+"TSPOS2.csv", index=False)



