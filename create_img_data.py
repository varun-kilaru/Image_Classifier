import cv2

import numpy as np

import os

from random import shuffle

from tqdm import tqdm

import tensorflow as tf

import matplotlib.pyplot as plt


# PATHS TO TRAININGDATA, TESTINGDATA, PREDICTDATA
train_dir = '/home/kilarubrothers/Desktop/NeuralNets/train_dataset'
test_dir = '/home/kilarubrothers/Desktop/NeuralNets/validation_dataset'
predict_dir = '/home/kilarubrothers/Desktop/NeuralNets/predict_dataset'


# FUNCTION TO DEFINE LABLES
def one_hot_label(img):
	global ohl
	label = img.split('.')[0]
	if(label == 'zero'):
		ohl = np.array([1,0,0,0,0,0])
	elif(label == 'one'):
		ohl = np.array([0,1,0,0,0,0])
	elif(label == 'two'):
		ohl = np.array([0,0,1,0,0,0])
	elif(label == 'three'):
		ohl = np.array([0,0,0,1,0,0])
	elif(label == 'four'):
		ohl = np.array([0,0,0,0,1,0])
	elif(label == 'five'):
		ohl = np.array([0,0,0,0,0,1])
	# print(label,'--->',ohl)
	return ohl


# FUNCTION TO CREATE TRAININGDATA
def create_train_data():
	print('\n')
	train_data = []
	for i in tqdm(os.listdir(train_dir)):
		path = os.path.join(train_dir, i)
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		# print(img.shape)
		train_data.append( [ np.array(img), one_hot_label(i) ] )
	shuffle(train_data)
	print(len(train_data))
	np.save("training_data.npy",train_data)
	print('Done creating train data....')
	print('\n')
	# return train_data


# FUNCTION TO CREATE TESTINGDATA
def create_test_data():
	test_data = []
	for i in tqdm(os.listdir(test_dir)):
		path = os.path.join(test_dir, i)
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		# print(img.shape)
		test_data.append( [ np.array(img), one_hot_label(i) ] )
	shuffle(test_data)
	print(len(test_data))
	np.save("testing_data.npy",test_data)
	print('Done creating test data....')
	print('\n')
	# return test_data


# FUNCTION TO CREATE PREDICTDATA
def create_predict_data():
	predict_data = []
	for i in tqdm(os.listdir(predict_dir)):
		path = os.path.join(predict_dir, i)
		actual_label = i.split('.')[0]
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img,(100,100))
		# print(img.shape)
		# plt.imshow(img,cmap='gray')
		# plt.show()
		predict_data.append( [ np.array(img), actual_label ] )
	# shuffle(predict_data)
	print(len(predict_data))
	return predict_data


# CALLING THE CREATE_TRAIN_DATA(), CREATE_TEST_DATA()
# train_data = 
create_train_data()
# test_data = 
create_test_data()
# predict_data = create_predict_data()