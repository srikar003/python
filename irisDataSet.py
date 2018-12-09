import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from IPython.display import Image
import graphviz #package for visualizing the data (for this u need to install using "conda" cmd::conda install python-graphviz)
iris=load_iris()

#feature_names represents all the headings of the features in the dataset
print(iris.feature_names)

#target_names represents the labels in the dataset(output values)
print(iris.target_names)

#values of iris dataset are available in the "data" for features and "target" for labels
print('data-->',iris.data[0],'target-->',iris.target[0])

#We will remove one element from each type of label in irisDataset and will use for testing
test_idx=[2,105,102]

#trainingData (we will remove the elements of testing in dataset of iris)
train_data=np.delete(iris.data,test_idx,axis=0) #axis represent rows=>0 and columns=>1 
train_target=np.delete(iris.target,test_idx)

#testingData (we will use the removed elements for tesing)
test_data=iris.data[test_idx]
test_target=iris.target[test_idx]

clf=tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

print("resulted output:",clf.predict(test_data))
print("original output:",test_target)
print(len(train_data))

#ToVisualize the tree
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=iris.feature_names,class_names=iris.target_names,filled=True, rounded=True,special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("iris");