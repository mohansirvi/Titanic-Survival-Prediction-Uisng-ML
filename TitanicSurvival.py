#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#Load the data
titanic = sns.load_dataset('titanic')
#Print the first 10 rows of data
titanic.head(10)


# In[3]:


#Count the number of rows and columns in the data set
titanic.shape


# In[4]:


#Get some statistics from our data set, count, mean standard deviation etc.
titanic.describe()


# In[5]:


#Get a count of the number of survivors 
titanic['survived'].value_counts()


# In[6]:


#Visualize the count of number of survivors
sns.countplot(titanic['survived'],label="Count")


# In[7]:


# Visualize the count of survivors for columns 'who', 'sex', 'pclass', 'sibsp', 'parch', and 'embarked'
cols = ['who', 'sex', 'pclass', 'sibsp', 'parch', 'embarked']

n_rows = 2
n_cols = 3

#Number of rows/columns of the subplot grid and the figure size of each graph
#NOTE: This returns a Figure (fig) and an Axes Object (axs)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.2,n_rows*3.2))

for r in range(0,n_rows):
    for c in range(0,n_cols):  
        
        i = r*n_cols+ c #index to go through the number of columns       
        ax = axs[r][c] #Show where to position each subplot
        sns.countplot(titanic[cols[i]], hue=titanic["survived"], ax=ax)
        ax.set_title(cols[i])
        ax.legend(title="survived", loc='upper right') 
        
plt.tight_layout()   #tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area


# In[8]:


#Look at survival rate by sex
titanic.groupby('sex')[['survived']].mean()


# In[9]:


#Look at survival rate by sex
titanic.groupby('sex')[['survived']].mean()


# In[10]:


#Look at survival rate by sex and class
titanic.pivot_table('survived', index='sex', columns='class')


# In[11]:


#Look at survival rate by sex and class visually
titanic.pivot_table('survived', index='sex', columns='class').plot()


# In[12]:


#Plot the survival rate of each class.
sns.barplot(x='class', y='survived', data=titanic)


# In[13]:


#Look at survival rate by sex, age and class
age = pd.cut(titanic['age'], [0, 18, 80])
titanic.pivot_table('survived', ['sex', age], 'class')


# In[15]:


#Plot the Prices Paid Of Each Class
plt.scatter(titanic['fare'], titanic['class'],  color = 'purple', label='Passenger Paid')
plt.ylabel('Class')
plt.xlabel('Price / Fare')
plt.title('Price Of Each Class')
plt.legend()
plt.show()


# In[16]:


#Count the empty (NaN, NAN, na) values in each column
titanic.isna().sum()


# In[17]:


#Look at all of the values in each column & get a count
for val in titanic:
  print(titanic[val].value_counts())
  print()


# In[18]:


#DROP REDUNDENT COLUMNS & REMOVE EMPTY ROWS
#embark_town = embarked
#alive = survived
#class = pclass

#alone = (sibsp or parch) meaning if you have siblings/spouses or parents/children on board than you are not alone else you are
#adult_male = (male and age >= 18) meaning if you are a male age 18 or older than true else false, same goes for the who column which tracks only adult males, adult females, and children
#who = (Males age >= 18, Females age >= 18, children age < 18)

#deck missing 688 / 891 = 77.22% of the data



# Drop / remove the columns
titanic = titanic.drop(['deck', 'embark_town', 'alive', 'class', 'alone', 'adult_male', 'who'], axis=1)

#Drop/remove the rows with missing values
titanic = titanic.dropna(subset =['embarked', 'age'])

#Note: Could've used .fillna() to fill in missing values for age like with the average.


# In[19]:


#Count the NEW number of rows and columns in the data set
titanic.shape


# In[20]:


#Look at the data types to see which columns need to be transformed / encoded to a number
titanic.dtypes


# In[21]:


#Print the unique values in the columns
print(titanic['sex'].unique())
print(titanic['embarked'].unique())


# In[22]:


#Encoding categorical data values (Transforming object data types to integers)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

#Encode sex column
titanic.iloc[:,2]= labelencoder.fit_transform(titanic.iloc[:,2].values)
#print(labelencoder.fit_transform(titanic.iloc[:,2].values))

#Encode embarked
titanic.iloc[:,7]= labelencoder.fit_transform(titanic.iloc[:,7].values)
#print(labelencoder.fit_transform(titanic.iloc[:,7].values))

#Print the NEW unique values in the columns
print(titanic['sex'].unique())
print(titanic['embarked'].unique())


# In[23]:


#Look at the NEW data types
titanic.dtypes


# In[24]:


#Split the data into independent 'X' and dependent 'Y' variables
X = titanic.iloc[:, 1:8].values #Notice I started from index  1 to 7, essentially removing the first column
Y = titanic.iloc[:, 0].values #Get the target variable


# In[25]:


# Split the dataset into 80% Training set and 20% Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[26]:


# Scale the data to bring all features to the same level of magnitude
# This means the data will be within a specific range for example 0 -100 or 0 - 1

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[27]:


#Create a function within many Machine Learning Models
def models(X_train,Y_train):
  
  #Using Logistic Regression Algorithm to the Training Set
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state = 0)
  log.fit(X_train, Y_train)
  
  #Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
  knn.fit(X_train, Y_train)

  #Using SVC method of svm class to use Support Vector Machine Algorithm
  from sklearn.svm import SVC
  svc_lin = SVC(kernel = 'linear', random_state = 0)
  svc_lin.fit(X_train, Y_train)

  #Using SVC method of svm class to use Kernel SVM Algorithm
  from sklearn.svm import SVC
  svc_rbf = SVC(kernel = 'rbf', random_state = 0)
  svc_rbf.fit(X_train, Y_train)

  #Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
  from sklearn.naive_bayes import GaussianNB
  gauss = GaussianNB()
  gauss.fit(X_train, Y_train)

  #Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
  tree.fit(X_train, Y_train)

  #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)
  
  #print model accuracy on the training data.
  print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
  print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
  print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
  print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
  print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
  print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
  print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  
  return log, knn, svc_lin, svc_rbf, gauss, tree, forest


# In[28]:


#Get and train all of the models
model = models(X_train,Y_train)


# In[29]:


#Show the confusion matrix and accuracy for all of the models on the test data
#Classification accuracy is the ratio of correct predictions to total predictions made.
from sklearn.metrics import confusion_matrix
for i in range(len(model)):
  cm = confusion_matrix(Y_test, model[i].predict(X_test))

  #extracting true_positives, false_positives, true_negatives, false_negatives
  TN, FP, FN, TP = confusion_matrix(Y_test, model[i].predict(X_test)).ravel()

  print(cm)
  print('Model[{}] Testing Accuracy = "{} !"'.format(i,  (TP + TN) / (TP + TN + FN + FP)))
  print()# Print a new line


# In[30]:


#Get the importance of the features
forest = model[6]
importances = pd.DataFrame({'feature':titanic.iloc[:, 1:8].columns,'importance':np.round(forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances


# In[31]:


#Visualize the importance
importances.plot.bar()


# In[32]:


#Print Prediction of Random Forest Classifier model
pred = model[6].predict(X_test)
print(pred)

#Print a space
print()

#Print the actual values
print(Y_test)


# In[33]:


# Given the data points would I have survived ? 
# Most likely I would've been in 3rd class (pclass = 3), Im a male (sex = 1), age is older than 18 (age = 21), no siblings onboard (sibsp = 0), 
#no parents or children (parch =0), fare the minimum price (fare = 0), embarked queens town = (embarked =1)
my_survival = [[3,1,21,0, 0, 0, 1]]

#uncomment to see all of the models predictions
#for i in range(len(model)):
#  pred = model[i].predict(my_survival)
#  print(pred)


#Print Prediction of Random Forest Classifier model
pred = model[6].predict(my_survival)
print(pred)

if pred == 0:
  print('Oh no! You didn’t make it')
else:
  print('Nice! You survived')


# In[ ]:




