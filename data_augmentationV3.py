import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

def random_invert_img(x, p=0.5):
  if  tf.random.uniform([]) < p:
    x = (255-x)
  else:
    x
  return x

def plotImages(images_arr):
    fig, axes = plt.subplots(3, 3, figsize=(8,8))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


gen = ImageDataGenerator(rescale = 1./255,rotation_range=45,shear_range=0.25,channel_shift_range=10, 
horizontal_flip=False, brightness_range=[0.1,3.0],fill_mode='reflect',preprocessing_function= (lambda x:random_invert_img(x, 0.5)))


image_path = '274.png' 

image = np.expand_dims(plt.imread(image_path),0)*255
print(image)

print(image.shape)

aug_iter = gen.flow(image)

aug_images = [next(aug_iter)[0].astype('float32') for i in range(12)]


