import numpy as np
import csv
import math	

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

for i in range(len(train_label_1)): #converting the data into float and writing it has array of size 1000 in train_label
	train_label.append(float(train_label_1[i][0]))

train = [] 

for i in range(len(train_data)): #combining the data and label into one array train and writing array of size (3*1000)
	train.append([train_data[i][0],train_data[i][1],train_label[i]])

x11 = [] #for data of 1st dimension and label 1
x21 = [] #for data of 2nd dimension and label 1

for i in range(len(train)):
	if train[i][2] == 1.0: #comparing whether the label is 1
		x11.append(train[i][0])
		x21.append(train[i][1])

p = float(len(x11))/float(len(train)) #calculating probability of label 1 training data

x12 = []  #for data of 1st dimension and label 2
x22 = []  #for data of 2nd dimension and label 2

for i in range(len(train)):
	if train[i][2] == -1.0: #comparing whether the label is -1
		x12.append(train[i][0])
		x22.append(train[i][1])

test_data = [[1,1],[1,-1],[-1,1],[-1,-1]] #testing data

for i in range(len(test_data)):
	print "For the ",i+1,"test case"
	p11= math.exp(-(pow(test_data[i][0]-np.mean(x11),2))/(2*(np.var(x11))))/(math.sqrt(2*np.pi*(np.var(x11)))) #calculating probablity for 1st dimension in case of label 1 of test data
	p21= math.exp(-(pow(test_data[i][1]-np.mean(x21),2))/(2*(np.var(x21))))/(math.sqrt(2*np.pi*(np.var(x21)))) #calculating probablity for 2nd dimension in case of label 1 of test data
        p12= math.exp(-(pow(test_data[i][0]-np.mean(x12),2))/(2*(np.var(x12))))/(math.sqrt(2*np.pi*(np.var(x12))))#calculating probablity for 1st dimension in case of label 2 of test data
	p22= math.exp(-(pow(test_data[i][1]-np.mean(x22),2))/(2*(np.var(x22))))/(math.sqrt(2*np.pi*(np.var(x22))))#calculating probablity for 2nd dimension in case of label 2 of test data
	y1=p11*p21*p #calculating probability of test data to be label 1
	y2=p12*p22*(1-p) #calculating probability of test data to be label -1
        print "Probability for label 1 :",y1
	print "Probability for label -1 :",y2
	if y1 >= y2 :
		print "Predicted Label :",1
	else:
		print "Predicted Label :",-1	
	









