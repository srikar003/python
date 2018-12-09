from sklearn.datasets import load_iris
iris=load_iris()

#set x for features and y for labels
x=iris.data
y=iris.target

#split irisdata into two halfs, one for testing and other for training
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=.5)

#create classifier
# from sklearn import tree
# classifier=tree.DecisionTreeClassifier();

#another approach to create classifier
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier();


#train the classifier
classifier.fit(x_train,y_train)

#find the predictions with the testdata
predictions_testData= classifier.predict(x_test)
print(predictions_testData)

#compare the predictions with the labels of test data
from sklearn.metrics import accuracy_score
print(accuracy_score(predictions_testData,y_test))

#compare the values in bargraph
import matplotlib.pyplot as plt
plt.hist([predictions_testData,y_test],color=['black','blue'])
plt.show()

# plt.plot(predictions_testData,color='blue')
# plt.show()