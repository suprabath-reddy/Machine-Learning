import numpy as np
import matplotlib.pyplot as plt

N = input("No. of training samples - ")

input1 = np.random.rand(N)
input2 = np.random.rand(N)
y = np.zeros(N)

# if value is between 0 and 0.5, we take it has 0 otherwise 1.
# output using XOR for training input.
for i in range(0,N):
        if ((input1[i] >= 0.5) and (input2[i]) < 0.5):
                y[i] = 1
        elif ((input1[i] < 0.5) and (input2[i]) >= 0.5):
                y[i] = 1

#Adding noise to label
mean = 0
std = 0.05

y += np.random.normal(mean, std, N)

bias = np.random.rand(2)
weight = np.random.rand(6) #4 weights for perceptron and other two for output

weight_updated = np.zeros((N,6))

learn_rate = 0.9

iteration = []
error_at_iteration = []

for j in range(0,10000): #We are going to have 10000 iterations.
        error = 0
        for i in range(0,N):
                net_h1 = weight[0]*input1[i] + weight[1]*input2[i] + bias[0]
                net_h2 = weight[2]*input1[i] + weight[3]*input2[i] + bias[0]
                h1 = 1/(1 + np.exp(-1*net_h1))
                h2 = 1/(1 + np.exp(-1*net_h2))
                net_output = weight[4]*h1 + weight[5]*h2 + bias[1]
                output = 1/(1 + np.exp(-1*net_output)) 
                error = error + (y[i]-output)**2

                #rate of change of error with respect to weights
                error_w1 = -1*(y[i]-output)*output*(1- output)*h1*(1 - h1)*input1[i]*weight[4]
                error_w2 = -1*(y[i]-output)*output*(1- output)*h1*(1 - h1)*input2[i]*weight[4]

                error_w3 = -1*(y[i]-output)*output*(1- output)*h2*(1 - h2)*input1[i]*weight[5]
                error_w4 = -1*(y[i]-output)*output*(1- output)*h2*(1 - h2)*input2[i]*weight[5]

                error_w5 = -1*(y[i]-output)*output*(1- output)*h1
                error_w6 = -1*(y[i]-output)*output*(1- output)*h2

                #Updated Weights
                weight_updated[i][0] = weight[0] - learn_rate*error_w1
                weight_updated[i][1] = weight[1] - learn_rate*error_w2
                weight_updated[i][2] = weight[2] - learn_rate*error_w3
                weight_updated[i][3] = weight[3] - learn_rate*error_w4
                weight_updated[i][4] = weight[4] - learn_rate*error_w5
                weight_updated[i][5] = weight[5] - learn_rate*error_w6

        iteration.append(j)
        error_at_iteration.append(error/(2*N))   

        for i in range(0,6):
                sum = 0
                for j in range(0,N):
                        sum = sum + weight_updated[j][i]
                sum  = sum/N
                weight[i] = sum

print "Final error - ",error_at_iteration[9999]

#Plotting the Error aginst number of iterations
plt.figure(1)
plt.title('Back Propogation at N = ' + str(N))
plt.plot(iteration,error_at_iteration)
plt.xlabel("No. of Iterations")
plt.ylabel("Error")
plt.show()  		
