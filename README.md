
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# **Behavioral Cloning** 

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image0]: ./images/pilotnet.png "PilotNet"
[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/center_2017_11_19_04_22_15_285.jpg "Center"
[image3]: ./images/recovery-1.jpg "Recovery Image"
[image4]: ./images/recovery-2.jpg "Recovery Image"
[image5]: ./images/recovery-3.jpg "Recovery Image"
[image6]: ./images/original.jpg "Normal Image"
[image7]: ./images/flipped.png "Flipped Image"
[image8]: ./images/center_2017_11_19_04_22_15_285.jpg "Normal Image"
[image9]: ./images/center-masked.png "Making Image"
[image10]: ./images/center-cropped.png "Cropped Image"
[image11]: ./images/loss-20-no-dropout.png "loss"

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model I implemented for cloning the driver's behavior follow PilotNet which id suggested by "Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car" https://arxiv.org/pdf/1704.07911.pdf.

![alt text][image0]

The model consists of 5 convolutional layers followed by 3 fullu connected hidden layers. The output layer has one neuron which gives the angle of streering wheel. The first 3 convolutional layer use (5,5) kernels, while the following 2 convonlutional layers use (3,3) kernels.

#### 2. Attempts to reduce overfitting in the model

In order to fight against overfitting, dropout layers are added between fully connected layers with dropout rate 50%.

The code snippet to desctibe the model looks like the below;

```
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
```

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, which decays learning rate over iteration. The learning rate decay is 0 by default. I could change the decay and the initial learning rate value if the training performance (both speed and optimization) is poor. Fortunately, the training data with the above model does not give any good reason to change it.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and turning curves as smmothly as possible.

One important learning from this project. The initial training data includes frames where my simulation car goes to either right or left side of the road. Even the training data includes frames where the car goes out of the road. 

Why? I need to get training data which includes recovering, and I need first to the right or left or even out of the road. However, what the models learns is not only how to recover but also how to get off the road, which is NOT what I expect. To overcome this funny result, I excluded the video frames where the car moves to the right or left of the road, and out of the road. Dataset only includes frames of recovering. With that, the model knows how to recover while trying to keep itself in the middle of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the famous and simple ConvNet, LeNet. I thought this model might be appropriate because the model needs to predict the streering wheel's angel for a given image taken from the onboard camera. This can be easily achieved by having 1 neuron in the output layer, and make it regression not classification. For that, MSE (Mean Squared Error) is chosen as the cost function.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that the model give good prediction on unseen input data. This is done by changing neural network architecture from simple to complex (or deeper), and by applying dropout.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track but never goes back to the road. To improve the driving behavior in these cases, I re-gathered training data which do not include the unwanted driving pattern, driving to the edge of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with 5 convolutional layers and 3 fully connected layers. The first 2 layers are to normalize inputs and to drop input images.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

Also, check out the below model summary. Keras provide summary() function to desctibe the model with number of parameters for each layers,  output share, and the total number of parameters of the model. As seen below, the number of parameters of convolutional layers are smaller than one of fully connected layers.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
```

How to plot the model and to give summary from Keras? Very simple.

```
plot_model(model, to_file='model.png')
model.summary()
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover to the middle of the road. These images show what a recovery looks like starting from the right of the road to the midle:

![alt text][image3] Car is off the road

![alt text][image4] Car is recovering to the middle

![alt text][image5] Car is also positioned in the midele of the road

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would make the model robust because the track mostly consists of left curves. The model learns how to follow left curves good enough, but without enough data telling how to follow right curves the model will be confused on the left right curve. To overcome this, I could use a track having more right curve, but flipping images give the same. For example, here is an image that has then been flipped:

![alt text][image6] Original image

![alt text][image7] Flipped image (streering wheel angle also flipped)

Also, to reduce any noise all images are cropped to contain only road not tree or water which are not necessary for learning. Here is what it is done.

Taking a look at the orignal image, trees are at the top of the image, and the part of the car is at the bottom of the image.

![alt text][image8]

So, the blue boxes are the area of no-interest in training the model.

![alt text][image9]

Let's crop these area to have the images which are necessary for behavior cloning.

![alt text][image10]

The cropping is embedded into the network architecture rather than done seperately. In this way, it is not necessary to tell the user of this model to drop images manually before feeding data into the model for inference.

```
model.add(Cropping2D(cropping=((70,25),(0,0))))
```

After the collection process, I had 14,263 number of data points and each consists of imges from the center, left and right camera along with steering wheel angle. I then preprocessed this data by adjusting left and right images by adding and subtracting angles so that these images could be used along with the center images together. With that, the total number of data is 57,052.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped to determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by repeating 20 iterations. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image11]

Curious on how well it works? Here is the video recording from self-driving car using the model trained. [See the video clip.](./video.mp4)