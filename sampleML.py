from sklearn import tree #decisiontreeclassifier
features =[[140,1],[130,1],[150,0],[170,0]]
labels=[0,0,1,1]

#training classifier
clf = tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)
print(clf.predict([[160,0]]))