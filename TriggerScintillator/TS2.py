import pandas as pd

import numpy as np


#This file assumes that the columns in the .csv file containing the TS positional data are named x1, y1, x2, and y2


df = pd.read_csv("POS2.csv")


NR = len(df.axes[0])#The number of events in the file

x1 = [df['x1'][i] for i in range(NR)]#Saves the x1 coordinates of the TS hits

y1 = [df['y1'][i] for i in range(NR)]#Saves the y1 coordinates of the TS hits


x2 = [df['x2'][i] for i in range(NR)]#Saves the x2 coordinates of the TS hits

y2 = [df['y2'][i] for i in range(NR)]#Saves the y2 coordinates of the TS hits


YT1 = [] #This is where the converted y1 coordinate will be saved

YT2 = [] #This is where the converted y2 coordinate will be saved

for BB in y1:#This goes through the y1 values and sorts them into 24 different categories
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

for BB in y2:#This goes through the y2 values and sorts them into 24 different categories
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



XT1 = []#Saves the x1 coordinate that has been sorted into 8 different categories
XT2 = []#Saves the x2 coordinate that has been sorted into 8 different categories

for BX in x1:

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
for BX in x2:

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
P = {"x1": XT1,"y1":YT1,"x2":XT2, "y2":YT2}#Creates a dictionary of the data

PDF = pd.DataFrame(P)#Creates a Data Frame

pd.to_csv("TSPOS2.csv",P)#Saves the data as a .csv file



