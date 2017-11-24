import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D, MaxPooling2D, Activation, Dropout 
import matplotlib.pyplot as plt
from keras.utils import plot_model

# Reading driving data file
lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# a generator function to return a batch for training and validating
def generator(samples, batch_size=32):
	num_samples = len(samples)
	correction = 0.2

	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset: offset+batch_size]

			images = []
			angles = []

			for batch_sample in batch_samples:

				# center image 0, left image 1, right image 2
				for i in range(3):
					name = './data/IMG/' + batch_sample[i].split('/')[-1]
					image = cv2.imread(name)
					angle = float(batch_sample[3])
					
					# For images taken from the left camera, adjust angle by adding the correction value
					if (i == 1):
						angle += correction
					# For images taken from the right camera, adjust angle by substracting the correction value	
					elif (i == 2):
						angle -= correction

					images.append(image)
					angles.append(angle)

					# If images are from the center camera, augment the image by flipping images vertically
					# It is because the track is most left turns. Flipping images vertically, it give images for describing right turns
					if (i == 0):
						images.append(cv2.flip(image,1))
						angles.append(-angle)

			X_train = np.array(images)
			y_train = np.array(angles)

			yield sklearn.utils.shuffle(X_train, y_train)

# Define traing generator and validation generator using traing samples and validation samples
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32) 

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
model.add(Dropout(0.5))
model.add(Dense(1))

# Use Mean-Squared Error because the model is a regression
# Adam Optimization with the default learning rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, and decay = 0
model.compile(loss='mse', optimizer='adam')

# Print out the summary of the defined model, and visualize the model network
plot_model(model, to_file='model.png')
model.summary()

history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=5) 

# Save the trained model to a file
model.save('model.h5')

# Visualize the loss history of training data and validation data over epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.png')
