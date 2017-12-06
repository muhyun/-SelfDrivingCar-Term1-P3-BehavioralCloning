import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D, MaxPooling2D, Activation, Dropout 
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.utils import plot_model
import sys

# hyper-parameters
epoch=10
batch_size = 32
correction = 0.25
title = 'augmenting'
keep_rate = 0.01

if (len(sys.argv) == 3):
	base_dir = sys.argv[1]
	title = sys.argv[2]
else:
	base_dir = 'data'
	title = ''

# Reading driving data file
lines = []
with open('./' + base_dir + '/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

all_angles = []


# Generator function to return a batch for training and validating
def generator(samples, batch_size=32):
	num_samples = len(samples)
	#correction = 0.25
	first_run = 0
	
	while 1:
		sklearn.utils.shuffle(samples)
		first_run = first_run + 1
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset: offset+batch_size]

			images = []
			angles = []

			for batch_sample in batch_samples:
				# randomly choose images from center camera 0, left camera 1, right camera 2
				camera = np.random.randint(3)
				
				angle = float(batch_sample[3])
				
				name = './' + base_dir + '/IMG/' + batch_sample[camera].split('/')[-1]
				image = cv2.imread(name)

				# For images taken from the left camera, adjust angle by adding the correction value
				if (camera == 1):
					angle += correction
				# For images taken from the right camera, adjust angle by substracting the correction value	
				elif (camera == 2):
					angle -= correction

				# add image to dataset
				images.append(image)
				angles.append(angle)
				if first_run == 1:
					all_angles.append(angle)

				# flip image if the image is from center camera
				if (camera == 0):
					images.append(cv2.flip(image,1))
					angles.append(-angle)
					if first_run == 1:
						all_angles.append(angle)

			X_train = np.array(images)
			y_train = np.array(angles)

			yield sklearn.utils.shuffle(X_train, y_train)

# Define traing generator and validation generator using traing samples and validation samples
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size) 

# PilotNet architecture from "Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car"
# https://arxiv.org/pdf/1704.07911.pdf
# Input : image taken by the center camera
# Output : Steering wheel angle
model = Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# Use Mean-Squared Error because the model is a regression
# Adam Optimization with the default learning rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, and decay = 0
model.compile(loss='mse', optimizer='adam')

# Callback for checkpoint
filepath="model-{epoch:02d}-{val_loss:.3f}.h5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
callbacks_list = [checkpoint]

# Print out the summary of the defined model, and visualize the model network
plot_model(model, to_file='model.png')
model.summary()

history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size,epochs=epoch, callbacks=callbacks_list) 

# Save the trained model to a file
model.save('model.h5')

print('the number of images in the dataset : {}'.format(len(all_angles)))
# Visualize the loss history of training data and validation data over epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.png')

# clear plot
plt.gcf().clear()

# historical graph
plt.plot(all_angles)
plt.title('steering angle history diagram after augmenting - ' + title)
plt.ylabel('steering angle')
plt.xlabel('time')
plt.savefig('angles-hist-'+base_dir+'-aug.png')

# clear plot
plt.gcf().clear()

# gaussian dist
plt.hist(all_angles,bins=100)
plt.title('steering angle distribution diagram after augmenting - ' + title)
plt.ylabel('# of instances')
plt.xlabel('steering angle')
plt.savefig('angles-dist-'+base_dir+'-aug.png')