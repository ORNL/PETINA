from sklearn import datasets
from sklearn.model_selection import train_test_split

dataset = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

from diffprivlib.models import GaussianNB

clf = GaussianNB()
clf.fit(X_train, y_train)
clf.predict(X_test)
print("Test accuracy: %f" % clf.score(X_test, y_test))