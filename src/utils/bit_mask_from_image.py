from PIL import Image
import numpy as np
def mask_to_bits(image:Image):
    width = image.width
    height = image.height
    pixels = image.load()
    bits = np
    for index_row,row in enumerate(pixels):
        for index_colunm,item in enumerate(row):
            if item[0]!=0:
                pass


