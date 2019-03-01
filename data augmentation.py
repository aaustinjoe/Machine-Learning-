#!/usr+*/bin/env python389*-
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 15:23:22 2019

@author: austin
"""



import numpy as np
from numpy import *
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from PIL import Image
import os

#i hju78ikl.o/p=*-+l0'
listing = os.listdir('/home/austin/Desktop/input') 

for x in range(1,20):
     preview='/home/austin/Desktop/augmentated'
     datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
     fill_mode='nearest')
    
     img = Image.open('/home/austin/Desktop/input/m'+str(x)+'.jpeg')  
     img.resize((128,128))
     img.convert('L')
    
     x = img_to_array(img)  # creating a Numpy array with shape (3, 150, 150)
     x = x.reshape((1,) + x.shape) # converting to a Numpy array with shape (1, 3, 150, 150)
    
     i = 0
     for batch in datagen.flow(x,batch_size=1,save_to_dir= preview, save_prefix='Modi', save_format='JPEG'):
           i += 1
           if i > 20:
               break 