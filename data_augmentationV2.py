import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def random_invert_img(x, p=0.5):
  if  tf.random.uniform([]) < p:
    x = (255-x)
  else:
    x
  return x

def data_augmenter():
    '''
    Create a Sequential model composed of 6 layers of preprocessing
    Returns:
    tf.keras.Sequential
    '''
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(tf.keras.layers.RandomFlip('horizontal'))
    data_augmentation.add(tf.keras.layers.RandomFlip('vertical'))
    data_augmentation.add(tf.keras.layers.RandomRotation(factor=0.4, fill_mode="wrap"))
    data_augmentation.add(tf.keras.layers.RandomZoom(0.2,0.2))
    data_augmentation.add(tf.keras.layers.RandomContrast((0.2,0.75)))
    data_augmentation.add(tf.keras.layers.RandomBrightness((-0.35,0.35)))
    data_augmentation.add(tf.keras.layers.Lambda(lambda x: random_invert_img(x, 0.25)))

    return data_augmentation


# load the image
img = tf.keras.utils.load_img('tag.jpg')
# convert to numpy array
data = tf.keras.utils.img_to_array(img)
# expand dimension to one sample
samples = np.expand_dims(data, 0)

# Add the image to a batch.
image = tf.cast(samples, tf.float32)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  augmented_image = data_augmenter()(image)
  plt.imshow(augmented_image[0]/255)
  plt.axis("on")
plt.show()