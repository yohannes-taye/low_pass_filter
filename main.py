import numpy as np
from math import floor, ceil
from PIL import Image
from numpy import asarray
import math
import matplotlib.pyplot as plt

'''
A function to convert from an ndarray and save 
it to the specified location
'''
def save_image(img: np.ndarray, address):
    image = Image.fromarray(img)
    return image.save(address)

def get_window(full_image, location, window_size):
    """
    1. check if it goes out of bound 
    2. if not generate a blank window of requested size 
    3. iteratively map pixels in full image to window 
    """
    window = []

    start_x = location[0] - 1
    start_y = location[1] - 1
    img_size_x, img_size_y, _ = full_image.shape
    for x in range(3):
        for y in range(3):
            try:
                if start_x + x < img_size_x and start_y + y < img_size_y:
                    pixel = full_image[start_x + x][start_y + y][0]
                    
                    window.append(pixel)
                else:
                    window.append(1)
            except IndexError:

                print("___________________")
                print("Index out of bound")
                print("___________________")
                print("location: " + str(location))
                print("image size: " + str(full_image.shape))

            # window.append(full_image[start_x + x][start_y + y])
            # print("Appending " + str(full_image[start_x + x][start_y + y]))
    return window


def lpf(image):
    row, col, _ = image.shape
    new_image = np.zeros([row, col], dtype=np.uint8)
    lpf_kernel_matrix = np.array(
        [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])

    for x in range(row):
        for y in range(col):
            window = get_window(image, [x, y], 3)
            window = np.array(window).reshape((3, 3))
            lpf_value = np.sum(np.multiply(lpf_kernel_matrix, window))
            new_image[x][y] = lpf_value
    save_image(new_image, "./img/ohhhmyyyygodddd.jpg")

if __name__ == "__main__":
    image = Image.open("./img/new_image.jpeg").convert('LA')
    img = asarray(image)
    print(img.shape)
    lpf(img)
