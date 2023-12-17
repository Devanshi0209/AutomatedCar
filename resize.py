import numpy as np
import cv2
import os
arr=os.listdir('D:/WindowsNoEditor/PythonAPI/examples/indianroads')
for filename in arr:
	directory=r'D:\WindowsNoEditor\PythonAPI\examples\indianroads'
	os.chdir(directory)	
	image= cv2.imread(filename)
	resimage=cv2.resize(image,(320,240))
	directory=r'D:\WindowsNoEditor\PythonAPI\examples\indian_roads'
	os.chdir(directory)
	cv2.imwrite(filename,resimage)