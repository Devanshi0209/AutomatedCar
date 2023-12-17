import csv
import numpy as np
import os
directory=r'D:\WindowsNoEditor\PythonAPI\examples\none'
rows=[]
for filename in os.listdir(directory):
	imagepath='D:/WindowsNoEditor/PythonAPI/examples/none/'+filename
	rows.append([imagepath,np.array([0,0,0,1])])
with open('tl.csv', 'a', newline='') as f_object:
	csvwriter = csv.writer(f_object)
	csvwriter.writerows(rows)

		
		

'''
directory=r'D:\WindowsNoEditor\PythonAPI\examples\none'

for filename in os.listdir(directory):
	with open('tl.csv', 'w', newline='') as csvfile:
		csvwriter = csv.writer(csvfile)
		imagepath='D:/WindowsNoEditor/PythonAPI/examples/none/'+filename
		label=np.array([0,0,0,1])
		csvwriter.writerow([imagepath,label])
'''