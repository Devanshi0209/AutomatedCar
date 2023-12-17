import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from scipy import sparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tensorflow.keras.models import save_model,load_model
from imgaug import augmenters as iaa
import dask.array as da
import csv
from collections import Counter
import tensorflow as tf

def imageandsteering(data):
	image_processed = []
	steering = []
	for i in range(len(data)):
		indexed_data = data.iloc[i]
		center = indexed_data[0]
		image_processed.append(center.strip())
		steering.append(float(indexed_data[1]))
	image_paths = np.asarray(image_processed)
	steerings = np.asarray(steering)
	return image_paths, steerings
def preprocessv2(image):
	image=cv2.imread(image)
	image=image[80:220,150:320,:]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	image=image/255
	return image
def preprocess(image):
	image=mpimg.imread(image)
	image=image[100:200,:,:]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
	image=cv2.GaussianBlur(image,(3,3),0)
	image=cv2.resize(image,(200,66))
	image=image/255
	return image


def createnvdia():
	model=Sequential()
	model.add(Conv2D(filters=24,kernel_size=(5,5),strides=(2,2), input_shape=(66,200,3), activation='elu'))
	model.add(Conv2D(filters=36,kernel_size=(5,5),strides=(2,2), activation='elu'))
	model.add(Conv2D(filters=48,kernel_size=(5,5),strides=(2,2), activation='elu'))
	model.add(Conv2D(filters=64,kernel_size=(3,3), activation='elu'))
	model.add(Conv2D(filters=64,kernel_size=(3,3), activation='elu'))
	#model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100,activation='elu'))
	#model.add(Dropout(0.5))
	model.add(Dense(50,activation='elu'))
	#model.add(Dropout(0.5))
	model.add(Dense(10,activation='elu'))
	#model.add(Dropout(0.5))
	model.add(Dense(1))

	optimizer=Adam(lr=1e-4)
	model.compile(loss='mse',optimizer=optimizer)
	return model



def creatergbdata():
	columns=['imagepaths','steer']
	data=pd.read_csv('D:\WindowsNoEditor\PythonAPI\examples\steeringvalues.csv',names=columns)
	num_bins=25
	samples_per_bin=18000
	hist, bins = np.histogram(data['steer'], num_bins)
	center = (bins[:-1]+ bins[1:]) * 0.5
	plt.bar(center,hist,width=0.05)
	plt.show()
	print("balancing data")
	remove_list = []
	for j in range(num_bins):
  		list_ = []
  		for i in range(len(data['steer'])):
  			if data['steer'][i] >= bins[j] and data['steer'][i] <= bins[j+1]:
  				list_.append(i)
  		list_ = shuffle(list_)
  		list_ = list_[samples_per_bin:]
  		remove_list.extend(list_)
	data.drop(data.index[remove_list], inplace=True)
	#print("remaining")
	#print(len(data))
	hist, bins = np.histogram(data['steer'], num_bins)
	plt.bar(center,hist,width=0.05)
	plt.show()

	print("imageandsteering")
	imageinputs,steeringoutputs=imageandsteering(data)

	print("dask arrays and processing")
	
	Xtrain,Xtest,Ytrain,Ytest=train_test_split(imageinputs,steeringoutputs,test_size=0.2,random_state=6)
	#print(preprocess(Xtrain[0]))

	Xtrain=np.array(list(map(preprocess,Xtrain)),dtype='float16')
	Xtest=np.array(list(map(preprocess,Xtest)),dtype='float16')
	print(Xtrain[0])
	print(Xtrain.shape)

	print("started training")
	model=createnvdia()
	history = model.fit_generator(batch_generator(Xtrain, Ytrain, 100, 1),steps_per_epoch=300, epochs=10,validation_data=(Xtest,Ytest),validation_steps=200,verbose=1,shuffle = 1)
	history=model.fit(Xtrain,Ytrain,epochs=15,validation_data=(Xtest,Ytest),batch_size=100,verbose=1,shuffle=1)
	model.save('D:\WindowsNoEditor\PythonAPI\examples\save_model-rgbdata-nvdia-1output-balanced\saved_model')
def testprediction():
	model=load_model('D:\WindowsNoEditor\PythonAPI\examples\save_model-rgbdata-nvdia-1output-balanced\saved_model')
	IMAGE_DIR = r'D:\WindowsNoEditor\PythonAPI\examples\indian_roads'
	IMAGE_PATHS = []

	for file in os.listdir(IMAGE_DIR):
		if file.endswith(".jpeg"):
			IMAGE_PATHS.append(os.path.join(IMAGE_DIR, file))
	for image_path in IMAGE_PATHS:
		image2=cv2.imread(image_path)
		image=image2[100:200,:]
		image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
		image=cv2.GaussianBlur(image,(3,3),0)
		image=cv2.resize(image,(200,66))
		image=image/255
		image=np.array([image])
		prediction=model.predict(image)[0]
		print("STEER PREDICTION: ",end=" ")
		print(float(prediction[0]))
		cv2.imshow("input image",image2)
		cv2.waitKey(0)


def convertmodel():
	model=load_model('D:\WindowsNoEditor\PythonAPI\examples\save_model-rgbdata-nvdia-1output-balanced\saved_model')
	save_model(model,'D:\WindowsNoEditor\PythonAPI\examples\save_model-manualdata-pbfile\saved_model.pb')

if __name__=="__main__":
	testprediction()