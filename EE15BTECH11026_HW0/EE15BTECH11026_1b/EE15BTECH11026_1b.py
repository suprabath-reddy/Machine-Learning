import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

#Function for finding psudo-inverse
def inverse(x,N,M):
        arr = []
        for i in range(0,N):
            temp=[]
            for j in range(0,M):
		temp.append(x[i]**j) # creating x vector for each x by raising x to power j
 	    arr.append(temp)
        arr = np.matrix(arr) #converting array to matrix for easy calculation
	A = arr.transpose() * arr #finding x_tranpose * x
        return np.linalg.inv(A) * arr.transpose() #returning psudo inverse

def error(y,y1,N):
	sum=0
	for i in range(0,N):
		sum+=(y[i]-y1[i])**2
	return (sum/N)
		
#Number of training samples
N=1000

#Polynomial Model order
M=10

#Generate equispaced floats in the interval 
x=np.linspace(0,2*np.pi,N)

#Noise Parameters
mean = 0
std = 0.05

#Generate some numbers from the sine function
y=np.sin(x)

# Add noise
y+=np.random.normal(mean,std,N)

y1 = np.matrix(y)
y1 = y1.transpose()  #for calculation purpose

psudo_inverse = inverse(x,N,M);
w = psudo_inverse * y1 # finding weights
print w

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

y_actual = []
for i in range(0,N_test):
	y_actual.append(np.sin(x_test[i]))
 
print "Average error:"
print error(y_test,y_actual,N_test)

plt.title('Polynomial Model = 10 ; Training Data: 1000, and Testing Data = 100')
plt.plot(x,y,'ro',label="Training Data")
plt.plot(x_test,y_test,'b*',label="Testing Data")
plt.legend()
plt.show()




