import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataHelper import keyCheck
from imagegrab_fordata import grab_screen



w = [1,0,0,0,0,0]
a = [0,1,0,0,0,0]
d = [0,0,1,0,0,0]
wa =[0,0,0,1,0,0]
wd =[0,0,0,0,1,0]
nk =[0,0,0,0,0,1]

def keys_to_output(keys):

    output = [0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'W' in keys:
        output = w
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output


fileName='train_datav2.npy'

if os.path.isfile(fileName):
	print('File exists,loading previous data')
	training_data=list(np.load(fileName,encoding='latin1',allow_pickle=True))
else:
	print('File doesnt exist,starting fresh')
	training_data=[]

if __name__=="__main__":
	paused=False
	while True:
		if not paused:
			screen=grab_screen(region=(0,100,1500,940))
			screen=cv2.cvtColor(screen,cv2.COLOR_RGB2GRAY)
			screen=cv2.resize(screen,(160,120))
			keys=keyCheck()
			output=keys_to_output(keys)
			training_data.append([screen,output])
			if len(training_data)%1000==0:
				print(len(training_data))
				np.save(fileName,training_data)
			keys = keyCheck()
			if 'T' in keys:
				if paused:
					paused = False
					print('unpaused!')
					time.sleep(1)
				else:
					print('Pausing!')
					paused = True
					time.sleep(1)
