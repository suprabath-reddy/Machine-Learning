import numpy as np
import csv
import math

def euclidean_distance(train, test): #calculating distances and returning in sorted order
	distance = []
	for i in range(len(train)):
            dis = 0
	    for j in range(len(train[i])-1):
                a = []
		dis += pow((train[i][j] - test[j]), 2) #calculating euclidean distance
	        a.append(math.sqrt(dis)) #array for saving distance and label
            a.append(train[i][2])
            distance.append(a) #array for test case with distance and label size - 2*1000
        return sorting(distance)
          

def sorting(distance):  #bubble sort the array distance
    n = len(distance)
 
    for i in range(n):
        for j in range(0, n-i-1):
            if distance[j][0] > distance[j+1][0]:
                distance[j], distance[j+1] = distance[j+1], distance[j] #swapping
    return distance

def knn(train,test,k):
	train = euclidean_distance(train, test) #calling function
	label = 0
	for i in range(k):
		label+=train[i][1] 
        label = label/k #taking average of first k labels
 	
	if label>=0:
		return 1 

        if label<0:
		return -1	
		

train1 = []
with open('X.csv', 'r') as csvFile:  #reading the X data file and saving it in train1
    data = csv.reader(csvFile)
    for column in data:
        train1.append(column)
csvFile.close()

train_data = []

for i in range(len(train1[0])): #converting the data into float and writing it has array of size(2*1000)
	train_data.append([float(train1[0][i]),float(train1[1][i])])

train_label_1 = []
with open('Y.csv', 'r') as csvFile: #reading the Y data file and saving it in train_label_1
    data = csv.reader(csvFile)
    for row in data:
        train_label_1.append(row)
csvFile.close()

train_label = []

for i in range(len(train_label_1)):  #converting the data into float and writing it has array of size 1000 in train_label
	train_label.append(float(train_label_1[i][0]))

train = []

for i in range(len(train_data)): #combining the data and label into one array train and writing array of size (3*1000)
	train.append([train_data[i][0],train_data[i][1],train_label[i]])


test_data = [[1,1],[1,-1],[-1,1],[-1,-1]]

for i in range(len(test_data)): #testing each test case
	print "For the ",i+1,"test case"
	a = []
	for k in range(1,1001): #for each test case checking for different k values
	       a.append(knn(train,test_data[i],k)) #array for saving values in case of different k values

        for j in range(len(a)-1): #checking for k value where there is change in prediction	
		if a[j+1]!=a[j]:	
			print j #value of k after which there is change in preidction, optimal k
                        break	 	
        print "Label :",a[j] #label of each test case





