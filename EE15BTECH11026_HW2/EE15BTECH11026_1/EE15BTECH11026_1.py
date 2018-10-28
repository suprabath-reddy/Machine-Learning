import numpy as np
import csv
import cvxpy as cp
from numpy import genfromtxt
import matplotlib.pyplot as plt

train1 = []
with open('Xsvm.csv', 'r') as csvFile:  #reading the X data file and saving it in train1
    data = csv.reader(csvFile)
    for column in data:
        train1.append(column)
csvFile.close()

X = []

for i in range(len(train1)): #converting the data into float and writing it has array of size(2*1000)
	X.append([float(train1[i][0]),float(train1[i][1])])

train_label_1 = []
with open('ysvm.csv', 'r') as csvFile: #reading the Y data file and saving it in train_label_1
    data = csv.reader(csvFile)
    for row in data:
        train_label_1.append(row)
csvFile.close()

Y = []

for i in range(len(train_label_1)):  #converting the data into float and writing it has array of size 1000 in train_label
	Y.append(float(train_label_1[i][0]))

#Solving the optimization problem using cvxpy library
alpha = cp.Variable(len(X))
sum_x1 =0                 ## for sum(alpha*y*x1)
sum_alpha = 0                ##  for sum(alpha)
sum_x2 = 0                ## for sum(alpha*y*x2)
sum_alphay =0                 ##for sum(alpha*y)

for i in range(0,len(X)):
	sum_x1 += alpha[i]*Y[i]*X[i][0]
	sum_alpha += alpha[i]
	sum_x2 += alpha[i]*Y[i]*X[i][1]
	sum_alphay += alpha[i]*Y[i]

sum_alphaxy = (cp.square(sum_x1)+cp.square(sum_x2))/2
objective = cp.Maximize(sum_alpha-sum_alphaxy)
constraints = [alpha>=0, sum_alphay==0]                #Constraint is alpha is positive and sum(alpha*y) = 0
prob = cp.Problem(objective, constraints)              #Solving the Optimization problem
result = prob.solve()
alpha_mat = alpha.value                                #Assigning the optimal alpha values to a matrix

weights = np.zeros(2)

#Caluculation of vector w
for i in range(0,len(X)):
	weights[0] = weights[0] + alpha_mat[i]*Y[i]*X[i][0]
	weights[1] = weights[1] + alpha_mat[i]*Y[i]*X[i][1]
w0 = 0

#Calculation of w0
for i in range(0,len(X)):
	w0 = w0 + 1/Y[i] - weights[0]*X[i][0] - weights[1]*X[i][1]
w0 = w0/len(X)

#Checking the accuracy of hyperplane w.r.t Training Data set
count = 0
for i in range(0,len(X)):
        error = weights[0]*X[i][0] + weights[1]*X[i][1] + w0
        if(error > 0):
                label = 1.0
        else :
                label = -1.0
        if(label != Y[i]):
                count = count +1

print ("Accuracy is ",(len(X)-count)*100.0/len(X),"%")

print ('Hyperplane is ',weights[0],'x1 + ',weights[1],'x2 + ',w0)

#Predicting label for test data
test_data = [[2,0.5],[0.8,0.7],[1.58,1.33],[0.008,0.001]]

for i in range(0,len(test_data)):
	if ( weights[0]*test_data[i][0] + weights[1]*test_data[i][1] + w0 > 0 ):
		label = 1.0
	else :
		label = -1.0
	print ('The points (',test_data[i][0],',',test_data[i][1],') has label : ',label)

#Plotting the Input data clusters and the Test data points
label1x = []
label1y = []
label2x = []
label2y = []
test_data_x = [2,0.8,1.58,0.008]
test_data_y = [0.5,0.7,1.33,0.001]

for i in range(0,len(X)):
	if (Y[i] == 1):
		label1x.append(X[i][0])
		label1y.append(X[i][1])
	else :
		label2x.append(X[i][0])
		label2y.append(X[i][1])

plt.figure(1)
plt.title('Classification with SVM')
plt.plot(label1x,label1y,'r*',label = 'Cluster 1')
plt.plot(label2x,label2y,'y.',label = 'Cluster 2')
plt.plot(test_data_x,test_data_y,'g.',label = 'Test Data')
plt.legend(loc='upper right')
plt.show()
