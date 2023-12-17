import matplotlib.pyplot as plt
import cv2
def carla_preprocess(image):
	image=image[100:200,:,:]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
	image=cv2.GaussianBlur(image,(3,3),0)
	image=cv2.resize(image,(200,66))
	image=image/255
	return image
image=cv2.imread('dataset/out00745296.jpg')
plt.imshow(image)
plt.show()
image=cv2.resize(image,(512,512))
image=image[219:475,256:512,:]
plt.imshow(image)
plt.show()
