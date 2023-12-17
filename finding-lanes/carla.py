import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
def makecoordinates(image,line_parameters):
	slope,intercept=line_parameters

	y1=image.shape[0]
	y2=int(y1*3/5)
	x1=int((y1-intercept)/slope)
	x2=int((y2-intercept)/slope)
	return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
	left_fit=[]
	right_fit=[]
	if lines is None:
		return None
	for line in lines:
		x1,y1,x2,y2=line.reshape(4)
		parameters=np.polyfit((x1,x2),(y1,y2),1)
		slope,intercept=parameters
		if slope<0:
			left_fit.append((slope,intercept))
		elif slope==0:
			continue
		else:
			right_fit.append((slope,intercept))
	if len(left_fit)==0 or len(right_fit)==0:
		return np.array([])

	left_fit_average=np.average(left_fit,axis=0)
	right_fit_average=np.average(right_fit,axis=0)
	left_line=makecoordinates(image,left_fit_average)
	right_line=makecoordinates(image,right_fit_average)
	return np.array([left_line,right_line])



def display_lines(image,lines):
	slope1=0
	slope2=0
	
	if lines is not None: # if some hough lines were detected or grids were detected
		for line in lines: #iterated through the lines
			x1,y1,x2,y2=line.reshape(4) #since the line is a 2d array with 4 columns we reshape it
			try:
				cv2.line(image,(x1,y1),(x2,y2),(255,0,0),10) #cv2.line draws a line on lined_image with the coordinated x1y1x2y2 with the color blue and thicknees 10
			except:
				continue
		return image
	else:
		return image

def region_of_interest(image):
	height=image.shape[0] # image .shape has m,n,l along y x and breadth
	width=image.shape[1]
	#plt.imshow(image)
	#plt.show()
	triangle=np.array([[(50,220),(150,50),(250,220)]])
	#triangle=np.array([[(0,380), (width,380),(300,200)]])
	
	mask=np.zeros_like(image) # for an array of pixels with same dimensions as aaray but all pixel values are zero so it will be black
	cv2.fillPoly(mask,triangle,255)
	masked_image=cv2.bitwise_and(image,mask)
	return masked_image

def edge_detection(image):
	gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	cv2.imshow("result",gray) #shows the image in a tab called results
	cv2.waitKey(0) #shows the result image for specified amount of milliseconds- if given as 0 it shows till we press a key on keyboard
	blur=cv2.GaussianBlur(gray,(3,3),0)
	cv2.imshow("result",blur) #shows the image in a tab called results
	cv2.waitKey(0) #shows the result image for specified amount of milliseconds- if given as 0 it shows till we press a key on keyboard
	canny=cv2.Canny(blur,10,45) #gradient image and highlights the edges
	plt.imshow(canny) #shows the image in a tab called results
	plt.show() #shows the result image for specified amount of milliseconds- if given as 0 it shows till we press a key on keyboard
	return canny

start_time = time.time()


image= cv2.imread('carlaimage.jpg')
copy_i3_2=np.copy(image)
cv2.imshow("result",image) #shows the image in a tab called results
cv2.waitKey(0) #shows the result image for specified amount of milliseconds- if given as 0 it shows till we press a key on keyboard
canny=edge_detection(copy_i3_2)
cropped_image=region_of_interest(canny) #this is the cropped gradient image
cv2.imshow("result",cropped_image) #shows the image in a tab called results
cv2.waitKey(0) #shows the result image for specified amount of milliseconds- if given as 0 it shows till we press a key on keyboard
lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,60,np.array([]), minLineLength=30,maxLineGap=3) #the last two arguments represent the dimensions of the grids in the hough space and the grid with the maximum intersections will be voted as the ro theeta valueed line equation that best represents our data,  then we have the last argument as the threshold which is the minimum number of intersections a grid should have to be accepted 
averaged_lines=average_slope_intercept(copy_i3_2,lines)
lined_image=display_lines(copy_i3_2,lines)
cv2.imshow("result",lined_image) #shows the image in a tab called results
cv2.waitKey(0) #shows the result image for specified amount of milliseconds- if given as 0 it shows till we press a key on keyboard
