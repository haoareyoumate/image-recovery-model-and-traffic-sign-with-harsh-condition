import numpy as np
import cv2
from scipy.signal import convolve2d

def add_brightness(image,random_brightness_coefficient):    
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS    
    image_HLS = np.array(image_HLS, dtype = np.float64)     
    #random_brightness_coefficient = np.random.uniform()+0.5 ## generates value between 0.5 and 1.5    
    image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient ## scale pixel values up or down for channel 1(Lightness)    
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255    
    image_HLS = np.array(image_HLS, dtype = np.uint8)    
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB    
    
    return image_RGB

def add_rain(image):
    image = cv2.resize(image,(128,128))
    noise = np.random.normal(255./2, 255./10, (128,128))
    noise[noise<120] = noise[noise<120]//2
    size = 11 # generating the kernel 
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[:,(size-1)//2] = np.ones(size)/size
    noise = convolve2d(noise, kernel_motion_blur, mode='same')
    zoom = 2
    noise = noise[image.shape[0]//2 - (image.shape[0]//2)//zoom : image.shape[0]//2 + (image.shape[0]//2)//zoom   ,   image.shape[1]//2 - (image.shape[1]//2)//zoom : image.shape[1]//2 + (image.shape[1]//2)//zoom]
    noise = noise.astype(np.uint8)
    noise = cv2.resize(noise,dsize = (image.shape[1],image.shape[0]))
    noise = ((noise-noise.min())/1*255/(noise.max()-noise.min())).astype(np.uint8)
    noise = noise/255
    noise = noise.reshape([noise.shape[0],noise.shape[1],1])

    #setting color of the rain
    image_blur= cv2.blur(image,(75,75))
    image_blur= cv2.blur(image_blur,(45,45))
    drop_color = np.full(image.shape,255)
    drop_color = cv2.cvtColor(drop_color.astype(np.uint8),cv2.COLOR_BGR2HSV)
    image_blur = cv2.cvtColor(image_blur,cv2.COLOR_BGR2HSV)
    drop_color[:,:,0] = np.median(image_blur[:,:,0],axis = 0)
    drop_color[:,:,1] = drop_color[:,:,1]*0.6 + image_blur[:,:,1]*0.4
    drop_color[:,:,2] = drop_color[:,:,2]*0.4 + image_blur[:,:,2]*0.6
    drop_color = cv2.cvtColor(drop_color.astype(np.uint8),cv2.COLOR_HSV2BGR)
    image_rain_noise = (noise*drop_color + image*(1-noise))
    image_rain_noise = add_brightness(image_rain_noise.astype(np.uint8),0.8)
    image_rain_noise[:,:,2] = image_rain_noise[:,:,2]*0.9
    image_rain_noise = image_rain_noise.astype(np.uint8)
    image_rain_noise= cv2.blur(image_rain_noise,(3,3)) ## rainy view are blurry
    return image_rain_noise

    