Aim:
To write a python program to check whether a list contains a sublist
Algorithm:
●	Start the program
●	Using input Method getting the input of list and sublist from User
●	Checking if the list containing the sublist or not.
●	If list contain sublist the output will be “list2 is a subset of list1”
●	Else the output will be “list2 is Not a subset of list1”

Program 1:
if set(list2).issubset(set(list1)):
    print('list2 is a subset of list1')
else:
    print('list2 is not a subset of list1')
Output:
list1 element5 7 9 10
sublist elements7 9
list2 is a subset of list1

Program 2:
sub_list=False;
for i in range(0,len(list1)):
    j=0
    while((i+j)<len(list1) and j<len(list2) and list1[i+j]==list2[j]):
      j+=1
    if j==len(list2):
        sub_list=True;
        break;
if(sub_list):
    print("list2 is a subset of list1");
else:
     print("list2 is not a subset of list1");
Output:
list1 element5 7 9 10
sublist elements7 8
list2 is not a subset of list1
Program 3:
list=[1,2,3,4,5,6,7];
print("The given list is ",str(list));
sub_list=[1,4,3];
res=False;
for idx in range(len(list)-len(sub_list)+1):
    if(list[idx:idx+len(sub_list)]==sub_list):
       res=True;
print("The given sub_list is ",res)
Output:
The given list is  [1, 2, 3, 4, 5, 6, 7]
The given sub_list is  False
list1 element

Program 4:
for i in list2:
    if i in list1:
        count=True
    else:
        count=False

if count:
    print("list2 is a subset of list1");
else:
    print("list2 is a not a subset of list1");
Output:
list1 element5 6 78 9 13
sublist elements5 6
list2 is a subset of list1
Result:
Thus the program has been executed successfully.

2.Replace dictionary values with their average
Program:	
def dict_avg_val(list_items):
    for d in list_items:
        n1=d.pop('M1')
        n2=d.pop('M2')
        d['Marks[M1+M2]']=(n1+n2)/2
    return list_items
Student_list=[{'id':1,'name':"XXX",'M1':72,'M2':70},
              {'id':2,'name':"YYY",'M1':80,'M2':80},
              {'id':3,'name':"ZZZ",'M1':92,'M2':90}]
print(dict_avg_val(Student_list))
Output:
[{'id': 1, 'name': 'XXX', 'Marks[M1+M2]': 71.0}, {'id': 2, 'name': 'YYY', 'Marks[M1+M2]': 80.0}, {'id': 3, 'name': 'ZZZ', 'Marks[M1+M2]': 91.0}]

3.Performing manipulation of tuple element.

countries = ("Spain", "Italy", "India", "England", "Germany")
print("Before manipulating the countries will be:")
print(countries)
temp = list(countries)
print("Countries: ",temp)
temp.append("Russia") #add item
print("After append the countries will be: ",temp)
temp.pop(3) #remove item
print("After removing the item in countries: ",temp)
temp[2] = "Finland" #change item
print("Changing the item in countries: ",temp)
countries = tuple(temp)
print("After manipulating the countries will be:")
print(countries)

Output:
Before manipulating the countries will be:
('Spain', 'Italy', 'India', 'England', 'Germany')
Countries:  ['Spain', 'Italy', 'India', 'England', 'Germany']
After append the countries will be:  ['Spain', 'Italy', 'India', 'England', 'Germany', 'Russia']
After removing the item in countries:  ['Spain', 'Italy', 'India', 'Germany', 'Russia']
Changing the item in countries:  ['Spain', 'Italy', 'Finland', 'Germany', 'Russia']
After manipulating the countries will be:
('Spain', 'Italy', 'Finland', 'Germany', 'Russia')

4.Pandas and numpy to get the powers of an array values element-wise. First array elements raised to powers of second array element.


Program:	
import numpy as np
import pandas as pd
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
result_numpy = np.power(arr1, arr2)
print("Numpy power result:")
print(result_numpy)
print("Numpy ** operator result:")
print(arr1 ** arr2)
series1 = pd.Series(arr1.flatten())
series2 = pd.Series(arr2.flatten())
result_pandas = series1.pow(series2)
print("Pandas power result:")
print(result_pandas)

Output:
Numpy power result:
[[  1   4  27]
 [256 3125 7776]]
Numpy ** operator result:
[[  1   4  27]
 [256 3125 7776]]
Pandas power result:
0       1
1       4
2      27
3     256
4    3125
5    7776
dtype: int64



5. Check whether a string is panagram or not

Program:	
import string

def is_pangram(sentence):
    alphabet = set(string.ascii_lowercase)
    # Convert all characters in the sentence to lowercase and ignore non-alphabet characters
    letters = set(char for char in sentence.lower() if char.isalpha())
    return letters >= alphabet

6.k-fold cross validation Algorithm.

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
X, y = datasets.load_iris(return_X_y=True)
clf = DecisionTreeClassifier(random_state=42)
k_folds = KFold(n_splits = 5)
scores = cross_val_score(clf, X, y, cv = k_folds)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

Output:
Cross Validation Scores:  [1.         1.         0.83333333 0.93333333 0.8       ]
Average CV Score:  0.9133333333333333
Number of CV Scores used in Average:  5

7.To read each row a given CSV file and print a list of strings.

Program:
import csv
with open('File1.csv', newline='') as csvfile:
 data = csv.reader(csvfile, delimiter=' ', quotechar='|')
for row in data:
    print(', '.join(row))
File1.csv:
RollNo,Name,Dept
23mca001,Abinaya,MCA
23mca002,Amuthavalli,MCA
23mca003,AnupShankar,MCA
23mca004,Ashika,MCA
23mca005,Bhuvaneshwari,MCA
Output:
RollNo,Name,Dept
23mca001,Abinaya,MCA
23mca002,Amuthavalli,MCA
23mca003,AnupShankar,MCA
23mca004,Ashika,MCA
23mca005,Bhuvaneshwari,MCA


8. k-mean Algorithm.

Program:
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

data = list(zip(x, y))
inertias = []

for i in range(1, len(data) + 1): 
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, len(data) + 1), inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


9.naïve bayes Algorithm.

Program:
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], 
              [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
Output:
Accuracy: 100.00%

10.Logistic Regression(scikit-learn).

Program:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

binary_indices = np.where(y != 2)
X = X[binary_indices]
y = y[binary_indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
X_train_2D = X_train[:, :2]
X_test_2D = X_test[:, :2]

model_2D = LogisticRegression()
model_2D.fit(X_train_2D, y_train)

h = .02
x_min, x_max = X_train_2D[:, 0].min() - 1, X_train_2D[:, 0].max() + 1
y_min, y_max = X_train_2D[:, 1].min() - 1, X_train_2D[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = model_2D.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c=y_train, edgecolor='k', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()

11.stacked generalization(stacking).
Program:
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
base_models = [
    ('decision_tree', DecisionTreeClassifier()),
    ('knn', KNeighborsClassifier()),
    ('svc', SVC(probability=True))
]
meta_model = LogisticRegression()
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

Output:
Accuracy: 100.00%
Confusion Matrix:
[[19  0  0]
 [ 0 13  0]
 [ 0  0 13]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        13
           2       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        


12.Support vector machine.

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import SVC 
import numpy as np
import pandas as pd
X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.40)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')
plt.title('Synthetic Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
clf = SVC(kernel='linear')
x = pd.read_csv("path/to/cancer.csv")  

if 'malignant' in x.columns and 'benign' in x.columns:
       y = x.iloc[:, 30].values  # or y = x['class_column_name'].values if you know the column name
    
    x_features = np.column_stack((x['malignant'], x['benign']))  # Ensure these column names are correct
    
    clf.fit(x_features, y)
        prediction1 = clf.predict([[120, 990]])
    prediction2 = clf.predict([[85, 550]])
    print(f"Prediction for [120, 990]: {prediction1}")
    print(f"Prediction for [85, 550]: {prediction2}")
else:
    print("Columns 'malignant' and 'benign' not found in the CSV file")
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)

plt.xlim(-1, 3.5)
plt.title('Decision Boundaries')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

12.Implement any deep learning python libraries.

import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)
model = Sequential([
    Embedding(10000, 64),
    SimpleRNN(64),
    Dense(46, activation='softmax')  # 46 classes for Reuters dataset
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")


