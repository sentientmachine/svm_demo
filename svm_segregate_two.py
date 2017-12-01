
import numpy as np

#mlpy requires emerge -avNu mlpy
import mlpy
import sys
#Import Support Vector Machine code from sklearn
from sklearn import svm

#The SVC module under SVC does classifications
from sklearn.svm import SVC

#matplot lib plots help us visualize what is going on
import matplotlib.pyplot as plt



#X are the training rows 2 dimensions, a list of lists containing x and y
#X = [[0, 0], [1, 1]]


np.random.seed(0)
mean1, cov1, n1 = [1, 5], [[1,1],[1,2]], 200  # 200 samples of class 1
x1 = np.random.multivariate_normal(mean1, cov1, n1)
y1 = np.ones(n1, dtype=np.int)

mean2, cov2, n2 = [2.5, 2.5], [[1,0],[0,1]], 300 # 300 samples of class -1
x2 = np.random.multivariate_normal(mean2, cov2, n2)
y2 = 0 * np.ones(n2, dtype=np.int)

X = np.concatenate((x1, x2), axis=0) # concatenate the 1 and -1 samples


#y is the target classifications 0 and 1
#y = [0, 1]
y = np.concatenate((y1, y2))


#Instantiate a new Support Vector Machine Classifier
clf = svm.SVC()

#fit the hyperplane between the clouds of data, should be fast as hell
clf.fit(X, y)

#specify config options, read the docs
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, 
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

#After being fitted, the model can then be used to predict new values:

#If we pass in unseen row x=2 and y=2 then the answer should be classification 1

production_point = [1., 2.5]

answer1 = clf.predict([production_point])

production_point = [-1., 5]

answer2 = clf.predict([production_point])

#reveal the training data input for reds and blues
#ob is "point blue"
#or is "point red"
#plt.plot(x1[:,0], x1[:,1], 'ob', x2[:,0], x2[:,1], 'or', production_point[0], production_point[1], 'ob', markersize=20)

plt.plot(x1[:,0], x1[:,1], 'ob', x2[:,0], x2[:,1], 'or', markersize = 5)

colormap = ['r', 'b']
color = colormap[answer1[0]]

plt.plot(production_point[0], production_point[1], 'o' + str(color), markersize=20)

#I want to draw the decision line


ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
import warnings
warnings.simplefilter("ignore", FutureWarning)
#The following method is buggy down to the core becuase it hurls warnings all over the place
#because it's contracts will change on every new version
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
plt.savefig("svm_segregate_two.py.png")
#plt.show()






#Given the weights W=svc.coef_[0] and the intercept I=svc.intercept_ , the decision boundary is the line
#y = a*x - b


sys.stdout.write("Answer1: " + str(answer1))
sys.stdout.write("Answer2: " + str(answer2))
sys.stdout.write("done")


exit()
W = clf._get_coef()[0]
I = clf.intercept_
print("W is: " + str(W))
print("I is: " + str(I[0]))


a = float(-W[0]) / float(W[1])
b =  float(I[0]) / float(W[1])
print("a is: " + str(a))
print("b is: " + str(b))

min_x = np.min(X[:,0], axis=0)
max_x = np.max(X[:,0], axis=0)
print(min_x)
print(max_x)

#exit()
xx = np.arange(min_x, max_x, 1.0)
#yy = - (W[0] * xx + b) / W[1] # separator line
print(xx)

yy =  a * xx - b
#
plot2 = plt.plot(yy, xx, 'og')
#
plt.show()



#SVMs decision function depends on some subset of the training data, called the 
#support vectors. Some properties of these support vectors can be found in 
#members support_vectors_, support_ and n_support:

#here is what is going on in the SVM's brain, it's just a definition of a hyperplane,
#in 2 dimensions, it's just a line, y=mx + b

# get support vectors
print(clf.support_vectors_)
#array([[ 0.,  0.],
#       [ 1.,  1.]])


# get indices of support vectors
#print(clf.support_)
#array([0, 1]...)

# get number of support vectors for each class
#print(clf.n_support_)
#array([1, 1]...)

