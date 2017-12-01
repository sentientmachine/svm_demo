
#Import Support Vector Machine code from sklearn
from sklearn import svm

#The SVC module under SVC does classifications
from sklearn.svm import SVC

import matplotlib
matplotlib.use('Agg')
#matplot lib plots help us visualize what is going on
import matplotlib.pyplot as plt


#X are the training rows 2 dimensions, x and y
X = [[0, 0], [1, 1]]

#y is the target classifications 0 and 1
y = [0, 1]


#reveal the training data input
plt.plot(X, y, 'ro')
plt.margins(1, 1)
#plt.show()


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
print("Answer: " + str(clf.predict([[2., 2.]])))
#array([1])  #yeah!


#SVMs decision function depends on some subset of the training data, called the 
#support vectors. Some properties of these support vectors can be found in 
#members support_vectors_, support_ and n_support:

#here is what is going on in the SVM's brain, it's just a definition of a hyperplane,
#in 2 dimensions, it's just a line, y=mx + b

# get support vectors
#print(clf.support_vectors_)
#array([[ 0.,  0.],
#       [ 1.,  1.]])


# get indices of support vectors
#print(clf.support_)
#array([0, 1]...)

# get number of support vectors for each class
#print(clf.n_support_)
#array([1, 1]...)

