1.List contains a sublist
a=list(input("Enter list values").split())
b=list(input("Enter sublist values").split())
if all(element in a for element in b):
 print("The sublist is present in the list")
else:
 print("The sublist is not present in the list")

2.Dictionary values with their arguments
marks = [ 
 {"m1":90,"m2":50}, 
 {"m1":50,"m2":50}, 
 {"m1":95,"m2":100}, 
] 
for mark in marks: 
 mark["avg"] = ( mark["m1"] + mark["m2"] ) / 2
print(marks)

3.Manipulation of tuple elements
a=(10,20,30)
print("Tuple:",a)
print("Type:",type(a))
print("Accessing index",a[1])
b=('a','b','c')
print('Concatenation',a+b)
c=('Python')*3
print("Repetition:",c)
print("Slice:",a[:2])
d=[1,2,3,4,5]
d=tuple(d)
print("Type Conversion:",type(d))

4.Power of an array values
import numpy as np 
import pandas as pd 
arr1 = np.array([[1, 2, 3], [4, 5, 6]]) 
arr2 = np.array([[1, 2, 3], [4, 5, 6]]) 
result_numpy = np.power(arr1, arr2)
print("Numpy power result:")
print(result_numpy) 
series1 = pd.Series(arr1.flatten()) 
series2 = pd.Series(arr2.flatten()) 
result_pandas = series1.pow(series2)
print("Pandas power result:")
print(result_pandas) 

5.String Pangram or not
import string
def is_pangram(sentence):
    alphabet="abcdefghijklmnopqrstuvwxyz"
    for char in alphabet:
        if char not in sentence.lower():  
            return False
    return True
sentence = "The quick brown fox jumps over the lazy dog"
if is_pangram(sentence):
    print("The string is a pangram")
else:
    print("The string is NOT a pangram")

6.K-Fold 
from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(random_state=42)
k_folds = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=k_folds)
print("Cross Validation Scores: ", scores)
print("Average CV Score: {:.2f}".format(scores.mean()))
print("Number of CV Scores used in Average: ", len(scores))

7.Read a csv file
mall.csv
CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
5,Female,31,17,40
6,Female,22,17,76
7,Female,35,18,6
8,Female,23,18,94
9,Male,64,19,3
10,Female,30,19,72

create the csv file and paste the path
import csv
with open(r'C:\Users\HP\Desktop\mall.csv', mode='r') as file:    #change the path
    csv_reader = csv.reader(file)
    for row in csv_reader:
        print(row)

8.k-Means Algorithm
mall.csv
CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
5,Female,31,17,40
6,Female,22,17,76
7,Female,35,18,6
8,Female,23,18,94
9,Male,64,19,3
10,Female,30,19,72

create a csv file and paste the path in the comment mentioned
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
X = pd.read_csv(r"C:\Users\HP\Desktop\mall.csv").iloc[:, [3, 4]].values   #change the path
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, 'o-')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(X)
colors = ['b', 'g', 'r', 'c', 'm']
for i in range(5):
    plt.scatter(X[y_pred == i, 0], X[y_pred == i, 1],
                s=100, c=colors[i], label=f'Cluster {i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=300, c='y', label='Centroids')
plt.title('Customer Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.legend()
plt.show()

9.Naive Bayes Algorithm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
x = np.array([[1,2],[2,3],[3,3],[4,5],[5,7],[6,8],[7,8],[8,9],[9,10],[10,11]])
y = np.array([0,0,0,0,1,1,1,1,1,1])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
model = GaussianNB().fit(x_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, model.predict(x_test))*100:.2f}%")

10.Logistic Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
X, y = load_iris(return_X_y=True)
X, y = X[y != 2], y[y != 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
X_train_2D = X_train[:, :2]
model_2D = LogisticRegression().fit(X_train_2D, y_train)
x_min, x_max = X_train_2D[:,0].min()-1, X_train_2D[:,0].max()+1
y_min, y_max = X_train_2D[:,1].min()-1, X_train_2D[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min,x_max,0.1), np.arange(y_min,y_max,0.1))
Z = model_2D.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train_2D[:,0], X_train_2D[:,1], c=y_train, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()

11.Stacked Generalization
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)     #test_size=0.5 to avoid 100% accuracy
base_models = [
 ('dt', DecisionTreeClassifier()),
 ('knn', KNeighborsClassifier()),
 ('svc', SVC(probability=True))
]
meta_model = LogisticRegression()
stack_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
stack_model.fit(X_train, y_train)
y_pred = stack_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

12.Support Vector Machine
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
X, Y = make_blobs(n_samples=500, centers=2, cluster_std=0.4, random_state=0)
clf = SVC(kernel='linear')
clf.fit(X, Y)
samples = [[0, 0], [2, 3]]
for s in samples:
    print(f"Prediction for {s}: {clf.predict([s])[0]}")
xfit = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
w = clf.coef_[0]
b = clf.intercept_[0]
yfit = - (w[0] / w[1]) * xfit - b / w[1]
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')
plt.plot(xfit, yfit, 'k-', lw=2)
plt.title("SVM Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()