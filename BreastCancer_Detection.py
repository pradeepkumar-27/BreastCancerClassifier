#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#Importing dataset
data = pd.read_csv('BreastCancer_Data.csv')
data.isna().sum()
data = data.dropna(axis=1)
print(data['diagnosis'].value_counts())
sb.countplot(data['diagnosis'],label="count")
#sb.pairplot(data,hue="diagnosis")
corr = data.corr()
plt.figure(figsize=(20,20))
#sb.heatmap(corr,annot=True,fmt='%')

#Split the data into features and labels
X = data.iloc[:,2:].values
y = data.iloc[:,1].values

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

#Splitting the data into 70% training set and 30% testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Creating a function of many Machine Learning models
def models(X_train,y_train):
    
    #Using Logistic Resgression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train,y_train)
    
    #Using Decision Tree algorithm
    from sklearn.tree import DecisionTreeClassifier
    tree =  DecisionTreeClassifier(criterion='entropy',random_state=0)
    tree.fit(X_train,y_train)
    
    #Using Random Forest Classification algorithm
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    forest.fit(X_train,y_train)
    
    #Using K-Nearest Neighbour algorithm
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=13,p=2)
    knn.fit(X_train,y_train)
    
    #using Support Vector Machines algorithm
    from sklearn.svm import SVC
    svm = SVC(kernel='linear',random_state=0)
    svm.fit(X_train,y_train)
    
    #Printing accuracy scores
    print('[0]Logistic Regression training accuracy score : ',log.score(X_train,y_train))
    print('[1]Decision Tree training accuracy score : ',tree.score(X_train,y_train))
    print('[2]Random Forest training accuracy score : ',forest.score(X_train,y_train))
    print('[3]KNN accuracy training score : ',log.score(X_train,y_train))
    print('[4]SVM accuracy training score : ',log.score(X_train,y_train))
    
    return log,tree,forest,knn,svm

#Calling the fuction models
model = models(X_train,y_train)

#Testing accuracy and other metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
  print('Model ',i)
  #Check precision, recall, f1-score
  print( classification_report(y_test, model[i].predict(X_test)) )
  #Another way to get the models accuracy on the test data
  print( accuracy_score(y_test, model[i].predict(X_test))*100)
  print()#Print a new line
  
#Since the testing accuracy of Logistic Regression is higher than that of any other trained models we use the Logistic Regression model for classification
#Print Prediction of Random Forest Classifier model
pred = model[0].predict(X_test)
print(pred)

print()

#Print the actual values
print(y_test)

#Confusion matrix visualization
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test,pred)
print(matrix)