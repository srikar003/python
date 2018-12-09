import numpy as np
from sklearn.datasets import load_wine
from sklearn import tree
from IPython.display import Image
import graphviz #package for visualizing the data (for this u need to install using "conda" cmd::conda install python-graphviz)
wine=load_wine()

#feature_names represents all the headings of the features in the dataset
print(wine.feature_names)

#target_names represents the labels in the dataset(output values)
print(wine.target_names)

#values of wine dataset are available in the "data" for features and "target" for labels
print('data-->',wine.data[0],'target-->',wine.target[0])

#We will remove one element from each type of label in wineDataset and will use for testing
test_idx=[2,52,102]

#trainingData (we will remove the elements of testing in dataset of wine)
train_data=np.delete(wine.data,test_idx,axis=0) #axis represent rows=>0 and columns=>1 
train_target=np.delete(wine.target,test_idx)

#testingData (we will use the removed elements for tesing)
test_data=wine.data[test_idx]
test_target=wine.target[test_idx]

clf=tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

print("resulted output:",clf.predict(test_data))
print("original output:",test_target)
print(len(train_data))
#ToVisualize the tree
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=wine.feature_names,class_names=wine.target_names,filled=True, rounded=True,special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("wine");