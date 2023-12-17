import cv2
import pandas as pd
import numpy as np
from collections import Counter
from random import shuffle
import matplotlib.pyplot as plt

train_data=np.load('train_data_balanced.npy',encoding='latin1',allow_pickle=True)

df=pd.DataFrame(train_data)


shuffle(train_data)
#print(len(train_data))

lefts = []
rights = []
forwards = []

'''
straight=[]
left=[]
right=[]
straightleft=[]
straightright=[]
nokeys=[]

for data in train_data:
	img=data[0]
	label=data[1]
	if label==[1,0,0,0,0,0]:
		straight.append([img,label])
	elif label==[0,1,0,0,0,0]:
		left.append([img,label])
	elif label==[0,0,1,0,0,0]:
		right.append([img,label])
	elif label==[0,0,0,1,0,0]:
		straightleft.append([img,label])
	elif label==[0,0,0,0,1,0]:
		straightright.append([img,label])
	elif label==[0,0,0,0,0,1]:
		nokeys.append([img,label])



straight=straight[:len(left)][:len(right)]
left=left[:len(straight)]
right=right[:len(straight)]
straightleft=straightleft[:len(straight)]
straightright=straightright[:len(straight)]
nokeys=nokeys[:len(straight)]

final_data=straight+left+right+straightleft+straightright+nokeys
'''

final_data=[]

final_final_data=[]
for i in train_data:
	img=i[0]
	img=img[:,15:80]
	blackpixels=np.sum(img==0)
	whitepixels=np.sum(img==255)
	almostwhitepixels=np.sum(img==254)	
	if whitepixels+blackpixels==6240 or whitepixels+blackpixels==3900 or whitepixels==3640 or almostwhitepixels==3640 or almostwhitepixels==3640 or almostwhitepixels==6240 or whitepixels==3835 or almostwhitepixels==3845 or (almostwhitepixels+whitepixels==3900) or (almostwhitepixels+whitepixels==3640) or blackpixels==2405:
		continue
	else:
		final_data.append(i)
		'''
		print("black")
		print(blackpixels)
		print("white")
		print(whitepixels)
		print("almostwhitepixels")
		print(almostwhitepixels)
		print("------------")
		cv2.imshow("res",img)
		cv2.waitKey(0)
'''


for data in final_data:
    img = data[0]
    choice = data[1]
    img=img[:,15:80]
    img=cv2.resize(img,(80,60))
    #plt.imshow(img)
    #plt.show()
    #cv2.imshow("res",img)
    #cv2.waitKey(0)

    if choice == [1,0,0]:
        lefts.append([img,choice])
    elif choice == [0,1,0]:
        forwards.append([img,choice])
    elif choice == [0,0,1]:
        rights.append([img,choice])



forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]
print("forwards")
print(len(forwards))
print("left")
print(len(lefts))
print("right")
print(len(rights))
final_final_data = forwards + lefts + rights



print("total")
print(len(final_final_data))
shuffle(final_final_data)
np.save('train_data_balanced_balanced.npy',final_final_data)


'''
np.save('training_data.npy', final_data)

shuffle(final_data)
print(len(final_data))
np.save('train_data_balancedv2.npy',final_data)
'''
