import pandas as pd
#This file assumes that the columns in the .csv file containing the TS positional data are named X1 and Y1

def Process_POS_1_event(input_file:str, output_directory:str):
    df = pd.read_csv(input_file)
    NR = len(df.axes[0])#The number of events in the file

    x = [df['X1'][i] for i in range(NR)]#Saves the x coordinates of the TS hits

    y = [df['Y1'][i] for i in range(NR)]#Saves the y coordinates of the TS hits

    YT = [] #This is where the converted y coordinate will be saved

    for BB in y:#This goes through the y values and sorts them into 24 different categories
        if BB <= -33.0:
            YT.append(0)
        elif -33.0<BB <= -30.0:
                YT.append(1)
        elif -30.0<BB <= -27.0:
            YT.append(2)
        elif -27.0<BB <= -24.0:
            YT.append(3)
        elif -24.0<BB <= -21.0:
            YT.append(4)
        elif -21.0<BB <= -18.0:
            YT.append(5)
        elif -18.0<BB <= -15.0:
            YT.append(6)
        elif -15.0<BB <= -12.0:
            YT.append(7)
        elif -12.0<BB <= -9.0:
            YT.append(8)
        elif -9.0<BB <= -6.0:
            YT.append(9)
        elif -6.0<BB <= -3.0:
            YT.append(10)
        elif -3.0<BB <= 0.0:
            YT.append(11)
        elif 0.0<BB <= 3.0:
            YT.append(12)
        elif 3.0<BB <= 6.0:
            YT.append(13)
        elif 6.0<BB <= 9.0:
            YT.append(14)
        elif 9.0<BB <= 12.0:
            YT.append(15)
        elif 12.0<BB <= 15.0:
            YT.append(16)
        elif 15.0<BB <= 18.0:
            YT.append(17)
        elif 18.0<BB <= 21.0:
            YT.append(18)
        elif 21.0<BB <= 24.0:
            YT.append(19)
        elif 24.0<BB <= 27.0:
            YT.append(20)
        elif 27.0<BB <= 30.0:
            YT.append(21)
        elif 30.0<BB <= 33.0:
            YT.append(22)
        elif BB >33.0:
            YT.append(23)

    #BX = 0

    XT = []#Saves the x coordinate that has been sorted into 8 different categories


    for BX in x:
        if BX <= -9.0:
            XT.append(0)
        elif -9.0 < BX <= -6.0:
            XT.append(1)
        elif -6.0 < BX <= -3.0:
            XT.append(2)
        elif -3.0 < BX <= 0:
            XT.append(3)
        elif 0.0 < BX <= 3.0:
            XT.append(4)
        elif 3.0 < BX <= 6.0:
            XT.append(5)
        elif 6.0 < BX <= 9.0:
            XT.append(6)
        elif BX > 9.0:
            XT.append(7)


    P = {"X1": XT,"Y1":YT}#Creates a dictionary of the data

    PDF = pd.DataFrame(P)#Creates a Data Frame

    PDF.to_csv(output_directory+"TSPOS1.csv", index=False)#Saves the data as a .csv file
