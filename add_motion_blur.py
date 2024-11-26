import cv2
import numpy as np

def motionblur(image):
    image = cv2.resize(image,(128,128))
    kernel_size = np.random.choice([14,15,16,17,18,19])
    kernel = np.zeros((kernel_size,kernel_size))
    kernel[kernel_size//2] = 1/kernel_size
    image = cv2.filter2D(image,ddepth=-1,kernel=kernel)
    return image