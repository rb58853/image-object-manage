import numpy as np


def inverse_array(arr):
    # Obtenemos las dimensiones del array
    rows, cols = arr.shape
    new_array = np.zeros((cols, rows))

    # Iteramos sobre los elementos del array original
    for index_row, row in enumerate(arr):
        for index_col, value in enumerate(row):
            new_array[index_col][index_row] = value

    return new_array
