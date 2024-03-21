import pandas as pd
import numpy as np

def process_value(num):
    try:
        return_value = int(num)
    except:
        return_value = -1
    finally:
        return return_value

def process_coordinates(xy:tuple, length:int):
    number_of_valid_coordinates = min(length - xy[0].count(-1), length - xy[1].count(-1))
    x_array = np.zeros(number_of_valid_coordinates, dtype=int)
    y_array = np.zeros(number_of_valid_coordinates, dtype=int)
    counter = 0
    for i in range(length):
        if xy[0][i] >= 0 and xy[1][i] >= 0:
            x_array[counter] = xy[0][i]
            y_array[counter] = xy[1][i]
            counter+=1 
    return (x_array, y_array)

        
