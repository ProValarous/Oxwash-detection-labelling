import tensorflow as tf
# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt




# load the image
img = tf.keras.utils.load_img('tag.jpg')
# convert to numpy array
data = tf.keras.utils.img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=[-200,200])
# prepare iterator
it = datagen.flow(samples, batch_size=1)

# generate samples and plot
for i in range(9):
	# define subplot
	plt.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	plt.imshow(image)
# show the figure
plt.show()


