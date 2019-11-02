import numpy as np

from keras.models import Sequential

from keras.layers import *

from keras.optimizers import *

from keras.callbacks import TensorBoard


IMG_SIZE = 100


# LOADING THE CREATED TRAINING & TESTING DATA
training_data = np.load('training_data.npy',allow_pickle=True)
testing_data = np.load('testing_data.npy',allow_pickle=True)


# Seperating the loaded data into image data & label data
train_img_data = np.array([i[0] for i in training_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
train_labels = np.array([i[1] for i in training_data])

test_img_data = np.array([i[0] for i in testing_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_labels = np.array([i[1] for i in testing_data])


# CONVOLUTION NEURAL NETWORK MODEL
model = Sequential()


# CONVOLUTION LAYER - 1
model.add(Conv2D(filters=64,kernel_size=5,strides=1,padding='same',activation='relu',input_shape=[IMG_SIZE, IMG_SIZE, 1]))
model.add(MaxPool2D(pool_size=5,padding='same'))


# CONVOLUTION LAYER - 2
model.add(Conv2D(filters=64,kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))


# CONVOLUTION LAYER - 3
model.add(Conv2D(filters=64,kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))


# FLATTENING THE DATA TO FEED INTO FULLY CONNECTED MODEL
model.add(Flatten())

tensorboard = TensorBoard(log_dir='./logs_dir', batch_size=10)


# DENSE LAYER - 1
model.add(Dense(100,activation='relu',input_shape=train_img_data.shape[1:]))


# DENSE LAYER - 2
# model.add(Dense(200,activation='relu'))
model.add(Dropout(rate=0.20))


# OUTPUT LAYER
model.add(Dense(6,activation='softmax'))


optimizer = SGD(lr=1e-4)


# COMPILING OUR MODEL AND FITTING THE TRAINDATA 
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=train_img_data,y=train_labels,epochs=30,batch_size=10,validation_data=(test_img_data,test_labels),callbacks=[tensorboard])
loss_metrics = model.evaluate(x=train_img_data,y=train_labels, batch_size=10, verbose=1)
print(loss_metrics)


# SAVE THE MODEL TO DISK
model.save('my_model.h5')
print('############################## Model saved to disk....')