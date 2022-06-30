 

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
 

digits = datasets.load_digits()


 

fig = plt.figure()
plt.imshow(digits.images[23],cmap = plt.cm.gray_r)
txt = "This is %d"%digits.target[23]
fig.text(0.1,0.1,txt)
plt.show()
 
digits.images[23]
 
x = 100 #length of training data set
X_train = digits.data[0:x]
Y_train = digits.target[0:x]

 
pred = 813
X_test = digits.data[pred]
print("X_test's real value is %d"%digits.target[pred])
 
def dist(x,y):
 return np.sqrt(np.sum((x-y)**2))
 
l = len(X_train)
distance = np.zeros(l) 
for i in range(l):
 distance[i] = dist(X_train[i],X_test)
min_index = np.argmin(distance)
print("Preditcted value is ",)
print(Y_train[min_index])
 
l = len(X_train)
no_err = 0
distance = np.zeros(l)
for j in range(1697,1797):
 X_test = digits.data[j]
 for i in range(l):
  distance[i] = dist(X_train[i],X_test)
 min_index = np.argmin(distance)
 if Y_train[min_index] != digits.target[j]:
  no_err+=1
print ("Total errors for train length = %d is %d"%(x,no_err))
 
#  Our testing data set has 100 examples. 
#  In the for loop it predicts the number for image at j index and 
# compares it with its actual value and then prints the total number of errors. 
# When x = 100 14/100 values are wrongly predicted and for x = 1696 2/100 values are wrongly predicted. 
# So our model predicts images with 98% accuracy.

 


 

