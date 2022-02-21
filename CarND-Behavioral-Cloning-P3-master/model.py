# reference : [Generators]
import os
import csv
import math


samples = []
with open ('./my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)
        
from sklearn.model_selection import train_test_split        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
#reference : [Using Multiple Cameras]
def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steer_angles = []
            for batch_sample in batch_samples:
                front_angle=float(batch_sample[3])
                revision = 0.2
                for camera_location in range(3):
                    path = './my_data/IMG/'+batch_sample[camera_location].split('/')[-1]
                    image = cv2.imread(path)
                    # change the color space BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #revise the angle
                    if camera_location == 0:
                        steer_angle = front_angle
                    elif camera_location == 1:
                        steer_angle = front_angle + revision
                    elif camera_location == 2:
                        steer_angle = front_angle - revision
   
                    # append image to images LIST
                    images.append(image)
                    steer_angles.append(steer_angle)
                    
                    # Flip Image & put angle of the opposite direction              
                    images.append(np.fliplr(image))
                    steer_angles.append(steer_angle*-1.0)
                
            # Trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(steer_angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# Set our batch size
batch_size = 32
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# Preprocess incoming data, centered around zero with small standard deviation 
from keras.models import Sequential
model = Sequential()

# Cropping
# reference : [Cropping Images in Keras]

# reference 2 : [Data Preprocessing]

from keras.layers import Cropping2D
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape =(160,320,3)))
from keras.layers import Input, Lambda
model.add(Lambda(lambda x: (x/255.0)-0.5))


# Add LAYERS
# using the Network similar to the "Even More Powerful Network"
from keras.layers.convolutional import Convolution2D

# reference : [Even More Powerful Network]

model.add(Convolution2D(24,5,5,subsample=(2,2)))
from keras.layers.core import Activation
model.add(Activation('relu'))
from keras.layers.core import Dropout
model.add(Dropout(0.1))
model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
from keras.layers.core import Flatten
model.add(Flatten())
from keras.layers.core import Dense
model.add(Dense(100)) 
from keras.layers.core import Dropout
model.add(Dropout(0.16))
model.add(Activation('relu'))
model.add(Dense(75)) 
model.add(Dropout(0.16))
model.add(Dense(50)) 
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dropout(0.16))
model.add(Activation('relu'))
model.add(Dense(1)) 







# Outputting Training and Validation Loss Metrics
model.compile(loss='mse',optimizer='adam')
# Fit data here
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), 
                    epochs = 7, verbose = 1)
# Save model here
model.save('model.h5')
# Keras method to print the model summary
model.summary()