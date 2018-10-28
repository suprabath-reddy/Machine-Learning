import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def inverse(x,N,M,variance_y,variance_w):
        arr = []
        for i in range(0,N):
            temp=[]
            for j in range(0,M):
		temp.append(x[i]**j) # creating x vector for each x by raising x to power j
 	    arr.append(temp)
        arr = np.matrix(arr) #converting array to matrix for easy calculation
	A = arr.transpose() * arr #finding x_tranpose * x
        return np.linalg.inv((A + (variance_y/variance_w)*np.identity(M, dtype = float)))* arr.transpose() #returning psudo inverse

def error(y,y1,N):
	sum=0
	for i in range(0,N):
		sum+=(y[i]-y1[i])**2
	return (sum/N)
		
#Number of training samples
N=100

#Polynomial Model order
M=5

#Generate equispaced floats in the interval 
x=np.linspace(0,2*np.pi,N)

#Noise Parameters
mean = 0
std = 0.05
variance_y = 1e-6   # y guassian variance
variance_w = 8e-3       # w guassian variance - alpha

#Generate some numbers from the sine function
y=np.sin(x)

# Add noise
y+=np.random.normal(mean,std,N)

y1 = np.matrix(y)
y1 = y1.transpose()  #for calculation purpose

psudo_inverse = inverse(x,N,M,variance_y,variance_w);
w = psudo_inverse * y1 # finding weights
print w

x_train = []
for i in range(0,N):
     temp=[]
     for j in range(0,M):
	temp.append(x[i]**j)    #creating x matrix
     x_train.append(temp)

x_train = np.matrix(x_train)          #writing as matrix

y_predict = x_train * w                   #predicting trained data
y_predict = np.array(y_predict.transpose())
 
y_train = []

for i in range(0,N):
	y_train.append(y_predict[0][i])
 
HH=np.random.normal(0,np.sqrt(variance_y),(100,1))

#This corresponds to test data 
N_test = 100
x_test = np.linspace(0, 2*np.pi, N_test)
x_test += np.random.normal(mean, std, N_test) #adding noise for testing data

x_test1 = []
for i in range(0,N_test):
     temp=[]
     for j in range(0,M):
	temp.append(x_test[i]**j)    # creating x matrix
     x_test1.append(temp)
x_test1 = np.matrix(x_test1)          #writing as matrix

y_test1 = x_test1 * w #calculating resultant y
y_test1 = np.array(y_test1.transpose())

y_test = []

for i in range(0,N_test):
	y_test.append(y_test1[0][i])

HH = np.array(HH.transpose()) 
HH_need = []

for i in range(0,N_test):
	HH_need.append(HH[0][i])

yn=[]
for i in range(0,N_test):
	yn.append(y_test[i] + HH_need[i]) #Finding new y_test by adding the variance calculated

y_actual = []
for i in range(0,N_test):
	y_actual.append(np.sin(x_test[i]))

print error(y_test,y_actual,N_test)
print error(yn,y_actual,N_test)

plt.title('Polynomial Model = 5 ; Training Data: 100, and Testing Data = 100')
plt.plot(x,y,'ro',label="Training Data")
plt.plot(x_test,y_test,'b*',label="First Prediction")
plt.plot(x_test,yn,'m+',label="Second Prediction")
plt.legend()
plt.show()




