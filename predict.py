from keras.models import *

import numpy as np

from create_img_data import create_predict_data

from tabulate import tabulate


# LOAD THE SAVED MODEL & PRINTING ITS SUMMARY
model = load_model('my_model.h5')
model.summary()
print('\n')


IMG_SIZE = 100


# CREATING THE PREDICTDATA
print('Creating data....')
predict_data = create_predict_data()
print('Creating data completed.')
print('\n')


# SEPERATING THE PREDICTDATA INTO IMAGE DATA & LABEL DATA
predict_img_data = np.array([i[0] for i in predict_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
actual_labels = np.array([i[1] for i in predict_data])


# MAKING THE PREDICTIONS
predictions = model.predict_classes(x=predict_img_data, batch_size=2, verbose=2)
n = predictions.shape[0]


# DISPLAYING THE PREDICTIONS MADE BY OUR MODEL
print('#########################>> PREDICTIONS <<#########################')
print('\n')


# DECLARATIONS
count = 0
c_w = [0]*n
predicted_labels = [' ']*n


# CONVERT OUR PREDICTIONS TO UNDERSTADALE CATEGORIES
CATEGORIES = ['zero', 'one', 'two', 'three', 'four', 'five']
for i in range(0,n):
	predicted_labels[i] = CATEGORIES[predictions[i]]
	if(CATEGORIES[predictions[i]] == actual_labels[i]):
		c_w[i] = 1
		count = count+1


# TO PRINT THE ANALYSIS 
print(tabulate({"PREDICTED": predicted_labels,"ACTUAL": actual_labels, "CORRECT/WRONG": c_w}, headers="keys"))
print('\n')
print('==>>> Total number of samples to be predicted : ',n)
print('==>>> Total correct answers : ',count)
print('==>>>Total incorrect answers :',n-count)