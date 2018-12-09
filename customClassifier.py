from scipy.spatial import distance

class customKNNClassifier():
    def euclid(self,a, b):
        return distance.euclidean(a, b)

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.nearest_featureLabel(row)
            predictions.append(label)
        return predictions

    def nearest_featureLabel(self, row):
        nearest_row = 0
        nearest_distance = self.euclid(row, self.x_train[0])
        for i in range(1, len(self.x_train)):
            distance = self.euclid(row, x_train[i])
            if nearest_distance > distance:
                nearest_distance = distance
                nearest_row=i
        return self.y_train[nearest_row]


from sklearn.datasets import load_iris
iris = load_iris()

#set x for features and y for labels
x = iris.data
y = iris.target

#split irisdata into two halfs, one for testing and other for training
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

#create classifier
classifier = customKNNClassifier()


#train the classifier
classifier.fit(x_train, y_train)

#find the predictions with the testdata
predictions_testData = classifier.predict(x_test)
print(predictions_testData)

#compare the predictions with the labels of test data
from sklearn.metrics import accuracy_score
print(accuracy_score(predictions_testData, y_test))

#compare the values in bargraph
import matplotlib.pyplot as plt
plt.hist([predictions_testData, y_test], color=['black', 'blue'])
plt.show()

# plt.plot(predictions_testData,color='blue')
# plt.show()