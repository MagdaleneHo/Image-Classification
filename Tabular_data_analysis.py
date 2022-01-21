
# Install auto-sklearn
# Reference: https://colab.research.google.com/github/vopani/fortyone/blob/main/notebooks/automl/tabular/Auto-Sklearn.ipynb#scrollTo=l4sGr3c3WDSW

!curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip3 install
!pip3 install auto-sklearn

# Check for updates and get latest version
!pip3 install auto-sklearn --upgrade

# Run this line and restart run time if error occur during autosklearn import
!pip3 install scikit-learn --upgrade

!pip install scikit-learn==0.24.2

# Install Pipeline Profiler
!pip3 install pipelineprofiler

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import pandas as pd 
import numpy as np 
import pickle
import autosklearn
import PipelineProfiler
from dateutil.relativedelta import relativedelta
from sklearn import metrics 
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from autosklearn.regression import AutoSklearnRegressor
from sklearn.metrics import r2_score, confusion_matrix, classification_report,  mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt

"""# 1. Read File"""

# Connect to your GDrive
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path

# Get folder path
data_path = Path("./drive/MyDrive/Colab Notebooks/Fast_Furious_Insured")

# Read csv file into DataFrame
ori_train_df = pd.read_csv(data_path/"train.csv")

"""# 2. Data Preparation

## 2.1 Drop Irrelevant Rows
"""

# Drop rows with condition 0 
# Do not need to predict those in good condition 
train_df = ori_train_df.drop(ori_train_df[ori_train_df['Condition'] == 0].index)
train_df.head(5)

# Remove invalid value in "Amount"
train_df.drop(train_df.index[(train_df["Amount"] <0)],axis=0,inplace=True)

"""## 2.2 Derive New Attributes"""

# Extract year from the Expiry_date 
train_df['Expiry_date'] = pd.to_datetime(train_df['Expiry_date'])
train_df['Expiry_year'] = pd.DatetimeIndex(train_df['Expiry_date']).year

# Create new attribute Year_difference
train_df['Year_difference'] = train_df['Expiry_year'] - 2021 
train_df.head(5)

# Convert "Insurance_company" to numeric representation (binary)
cat_vars = ['Insurance_company']
for var in cat_vars:
    cat_list = 'var'+'_'+var
    cat_list = pd.get_dummies(train_df[var], prefix=var)

    train_df1 = train_df.join(cat_list)
    train_df = train_df1
    

data_vars = train_df.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]

train_df =train_df[to_keep]

# View columns created
train_df.columns.values

""" ## 2.3 Drop Irrelevant Columns"""

# Drop unused attributes 
train_df = train_df.drop(columns=['Image_path', 'Expiry_date', 'Expiry_year', 'Condition'])
train_df.head(5)

"""## 2.4 Treat Missing Values

"""

# Checking for missing values
train_df.isnull().sum()

def impute_missing(df, option):
  # Option 1: Fill missing values with values from next row, fill with 0 if there are still missing values
  if option == 1:
    df = df.fillna(method='bfill', axis=0).fillna(0)

  # Option 2: Fill missing values with mean
  elif option == 2:
    df = df.fillna(df.mean())

  # Option 3: Fill missing values with 0
  elif option == 3:
    df = df.fillna(0)

  # Option 4: Remove rows with missing values
  elif option == 4:
    df = df.dropna(axis=0, how='any')

  return df

"""## This point onwards you can perform your specific data prep

1. Place it under your own modeling section (3.1/ 3.2/ 3.3)
2. It should not affect other modeling sections, so create a new df for it
3. Name them using m#_train_df (replace # according to your method, sample shown below)
4. The change attribute type can only be applied when there's no missing values, so run the chosen imputation method before running the line with astype(int)

*This cell to be deleted after done transfering codes*

# 3.0 Data Modeling

### 3.1 Continuous Target 

*   Decision Tree
*   Linear Regression
*   Gaussian Naive Bayes
*   K-Nearest Neighbors (KNN)

#### Preprocessing the variables
"""

# Current chosen imputation method
m1_train_df = impute_missing(train_df, 4)

# Convert all columns to int type
m1_train_df = m1_train_df.astype(int) 
print(m1_train_df.info())

"""#### Feature selection """

# Split the data into X & y

X = m1_train_df.drop('Amount', axis = 1).values
y = m1_train_df['Amount']

y = y.astype(int)

print(X.shape)
print(y.shape)
train_df.head()

# Run a Tree-based estimators 
dt = DecisionTreeClassifier(random_state=1, criterion = 'entropy')
dt.fit(X,y)

# Running Feature Importance

fi_col = []
fi = []

for i,column in enumerate(m1_train_df.drop('Amount', axis = 1)):
    print('The feature importance for {} is : {}'.format(column, dt.feature_importances_[i]))
    
    fi_col.append(column)
    fi.append(dt.feature_importances_[i])

#RFE process 
X = m1_train_df.loc[:, m1_train_df.columns != 'Amount']
y = m1_train_df['Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sel = RFE(DecisionTreeClassifier(),n_features_to_select = 10)
sel.fit(X_train, y_train)
sel.get_support()
features = X_train.columns[sel.get_support()]
print(features)

"""#### Modelling and evaluation """

# Split data into X and y

X = m1_train_df[features].values
y = m1_train_df['Amount']
print(X.shape)

"""Linear regression"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
LR_y_pred = lin_reg.predict(X_test)
print("The Testing Accuracy is: ", lin_reg.score(X_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, LR_y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, LR_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, LR_y_pred)))

"""Decision Tree"""

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.2)
dt =  DecisionTreeClassifier()
dt =  dt.fit(X_train,y_train)
dt_y_pred = dt.predict(X_test)
print('Accuracy:',metrics.accuracy_score(y_test,dt_y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, dt_y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, dt_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, dt_y_pred)))

"""Gaussion Naive Bayes"""

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)
gnb_y_pred = gnb.predict(X_test)
print("The Testing Accuracy is: ", gnb.score(X_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, gnb_y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, gnb_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, gnb_y_pred)))

"""K Nearest Neighbours (KNN)"""

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# K is user specified, check the best k: 
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.2)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)
print("The Testing Accuracy is: ",knn.score(X_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, knn_y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, knn_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, knn_y_pred)))

"""Saving the model"""

# save model
with open('gnb.pkl', 'wb') as f:
    pickle.dump(gnb, f)

# load model
with open('gnb.pkl', 'rb') as f:
    loaded_regressor = pickle.load(f)

"""### 3.2 Categorical Target


*   Logistic Regression
*   Support Vector Machine (SVM)
*   Decision Tree
*   Gaussian Naive Bayes
*   K-Nearest Neighbors (KNN)

#### Preprocessing the variables
"""

# Current chosen imputation method
m2_train_df = impute_missing(train_df, 2)

# Convert all columns to int type
m2_train_df = m2_train_df.astype(int) 
print(m2_train_df.info())

# Transform the skewed variables to log
m2_train_df['Cost_of_vehicle'] = m2_train_df['Cost_of_vehicle'].transform([np.log])
m2_train_df['Min_coverage'] = m2_train_df['Min_coverage'].transform([np.log])
m2_train_df['Max_coverage'] = m2_train_df['Max_coverage'].transform([np.log])

# Replace the infinity values to 0 when treating missing values of option 1 & 3
m2_train_df = m2_train_df.replace([np.inf, -np.inf], 0)

# Convert the amount into binary using the median as a break (less than 4048 = 0, more than 4048 = 1)
m2_train_df["Binary_Amount"] = (m2_train_df["Amount"] >= m2_train_df["Amount"].median()).astype(int)

# Drop unused attributes 
m2_train_df = m2_train_df.drop(columns=['Amount'])

m2_train_df.head(5)

"""#### Feature selection """

# Split the data into X & y

X = m2_train_df.drop('Binary_Amount', axis = 1).values
y = m2_train_df['Binary_Amount']

y = y.astype(int)

print(X.shape)
print(y.shape)
m2_train_df.head()

# Run a Tree-based estimators 
dt = DecisionTreeClassifier(random_state=1, criterion = 'entropy')
dt.fit(X,y)

# Running Feature Importance

fi_col = []
fi = []

for i,column in enumerate(m2_train_df.drop('Binary_Amount', axis = 1)):
    print('The feature importance for {} is : {}'.format(column, dt.feature_importances_[i]))
    
    fi_col.append(column)
    fi.append(dt.feature_importances_[i])

#RFE process 
X = m2_train_df.loc[:, m2_train_df.columns != 'Amount']
y = m2_train_df['Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sel = RFE(DecisionTreeClassifier(),n_features_to_select = 10)
sel.fit(X_train, y_train)
sel.get_support()
features = X_train.columns[sel.get_support()]
print(features)

"""### Modelling and evaluation"""

# Split data into X and y

X = m2_train_df[features].values
y = m2_train_df['Amount']
print(X.shape)

"""Logistic regression"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
log_reg = LogisticRegression(random_state=10, solver = 'lbfgs')
log_reg.fit(X_train, y_train)
log_reg.predict(X_train)
y_pred = log_reg.predict(X_train)
pred_proba = log_reg.predict_proba(X_train)
print("The Testing Accuracy is: ", log_reg.score(X_test, y_test))
print(classification_report(y_train, y_pred))

"""Support Vector Machine (SVM) - RBF kernel"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
svm_rbf.predict(X_train)
y_pred = svm_rbf.predict(X_train)
print("The Testing Accuracy is: ", svm_rbf.score(X_test, y_test))
print(classification_report(y_train, y_pred))

"""Decision Tree"""

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,test_size=0.2)
dt =  DecisionTreeClassifier()
dt =  dt.fit(X_train,y_train)
dt_y_pred = dt.predict(X_test)
print('Accuracy:',metrics.accuracy_score(y_test,dt_y_pred))
print(classification_report(y_train, y_pred))

"""Gaussian Naive Bayes"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.predict(X_train)
y_pred = gnb.predict(X_train)
print("The Testing Accuracy is: ", gnb.score(X_test, y_test))
print(classification_report(y_train, y_pred))

"""K Nearest Neigbours (KNN)"""

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# K is user specified, check the best k: 
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.2)
knn = KNeighborsClassifier(n_neighbors = 12)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
print("The Testing Accuracy is: ",knn.score(X_test, y_test))
print(classification_report(y_train, y_pred))

"""Saving the model"""

# save model
with open('svm_rbf.pkl', 'wb') as f:
    pickle.dump(svm_rbf, f)

# load model
with open('svm_rbf.pkl', 'rb') as f:
    loaded_regressor = pickle.load(f)

"""## 3.3 AutoML

Reference: Auto-Sklearn 2.0, https://automl.github.io/auto-sklearn/master/
"""

# Current chosen imputation method
m3_train_df = impute_missing(train_df, 4)

# Convert all columns to int type
m3_train_df = m3_train_df.astype(int) 
print(m3_train_df.info())

# Split the data into X & y
X = m3_train_df.drop('Amount', axis = 1).values
y = m3_train_df['Amount']

X = X.astype(int)
y = y.astype(int)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sklearn_regression_aml_5min= AutoSklearnRegressor(time_left_for_this_task=60*5)
sklearn_regression_aml_5min.fit(X_train, y_train, dataset_name='Insurance data')

y_pred = sklearn_regression_aml_5min.predict(X_test)

r2 = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
print(" R2: ", r2, "\n", "MAE:", mae, "\n", "RMSE: ", rmse)

show_modes_str=sklearn_regression_aml_5min.show_models()
sprint_statistics_str = sklearn_regression_aml_5min.sprint_statistics()

print(show_modes_str)
print(sprint_statistics_str)

profiler_data = PipelineProfiler.import_autosklearn(sklearn_regression_aml_5min)
PipelineProfiler.plot_pipeline_matrix(profiler_data)

# save model
with open('sklearn_regression_aml_5min.pkl', 'wb') as f:
    pickle.dump(sklearn_regression_aml_5min, f)

# load model
with open('sklearn_regression_aml_5min.pkl', 'rb') as f:
    loaded_regressor = pickle.load(f)

"""---

# **OLD codes to be deleted after transfer**

## 3.1 Data Transformation

Transformation of the variables:

```
Only LOG have output
Only some modelling algorithms improved with the transformed variables through log.
```
"""

#apply transformation to 'Cost_of_vehicle', 'Min_coverage' & 'Max_coverage'
trans1 = train_df['Max_coverage'].transform([np.log, np.exp, np.reciprocal, np.sqrt])

#Seeing which transformation is the best (ONLY *LOG* has output)
trans1.hist(bins=20)
plt.suptitle('Transformed Output')
plt.show()

#change age to the tranformation you have choosen
train_df['Cost_of_vehicle'] = train_df['Cost_of_vehicle'].transform([np.log])
train_df['Min_coverage'] = train_df['Min_coverage'].transform([np.log])
train_df['Max_coverage'] = train_df['Max_coverage'].transform([np.log])

"""Creating dummy variable:"""

# Extract the year from the date and return the year difference (cause it makes more sense?)
train_df['Expiry_date'] = pd.to_datetime(train_df['Expiry_date'])
train_df['Expiry_year'] = pd.DatetimeIndex(train_df['Expiry_date']).year
train_df['Year_difference'] = train_df['Expiry_year'] - 2021 
train_df.head(5)

# Drop unused variables 
train_df = train_df.drop(columns=['Image_path', 'Expiry_date', 'Expiry_year'])
train_df.head(5)

# Convert "Insurance_company" to numeric representation (binary)
cat_vars = ['Insurance_company']
for var in cat_vars:
    cat_list = 'var'+'_'+var
    cat_list = pd.get_dummies(train_df[var], prefix=var)

    train_df1 = train_df.join(cat_list)
    train_df = train_df1
    

data_vars = train_df.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]

new_train_df =train_df[to_keep]
new_train_df.columns.values
new_train_df = new_train_df.astype(int) #change all binary to int type
print(new_train_df.info())

# Split the data into X & y
X = new_train_df.drop('Amount', axis = 1).values
y = new_train_df['Amount']

y = y.astype(int)

print(X.shape)
print(y.shape)
train_df.head()

"""## 3.2 Feature selection """

# Run a Tree-based estimators (i.e. decision trees & random forests)
#dt = RandomForestClassifier(random_state=1, criterion = 'entropy')#, max_depth = 10)

dt = DecisionTreeClassifier(random_state=1, criterion = 'entropy')#, max_depth = 10)
dt.fit(X,y)

# Running Feature Importance
fi_col = []
fi = []

for i,column in enumerate(new_train_df.drop('Amount', axis = 1)):
    print('The feature importance for {} is : {}'.format(column, dt.feature_importances_[i]))
    
    fi_col.append(column)
    fi.append(dt.feature_importances_[i])

#RFE process 
X = new_train_df.loc[:, new_train_df.columns != 'Amount']
y = new_train_df['Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sel = RFE(DecisionTreeClassifier(),n_features_to_select = 10) #can change the n features, up to you
sel.fit(X_train, y_train)
sel.get_support()
features = X_train.columns[sel.get_support()]
print(features)

"""

---


## **Method 1: Continuous as Target (Amount)** 


 """

# Selecting the x and y
X = new_train_df[features].values
y = new_train_df['Amount']
print(X.shape)

# Linear regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
LR_y_pred = lin_reg.predict(X_test)
print("The Testing Accuracy is: ", lin_reg.score(X_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, LR_y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, LR_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, LR_y_pred)))

# Decision tree (accuracy = 0.07279693486590039)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,test_size=0.2)
dt =  DecisionTreeClassifier()
dt =  dt.fit(X_train,y_train)
dt_y_pred = dt.predict(X_test)
print('Accuracy:',metrics.accuracy_score(y_test,dt_y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, dt_y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, dt_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, dt_y_pred)))

# Gausian Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)
gnb_y_pred = gnb.predict(X_test)
print("The Testing Accuracy is: ", gnb.score(X_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, gnb_y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, gnb_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, gnb_y_pred)))

# K nearest neighbours 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# K is user specified , how do we know which K is the best to use ? 
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]); 

# I just do up to this part cause we can already see the highest accuracy from this chart is 0.0735 ...

# K NEAREST NEIGHBOURS

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.2)
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)
print("The Testing Accuracy is: ",knn.score(X_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, knn_y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, knn_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, knn_y_pred)))

"""---


## **Method 2: Categorical as Target (Binary_Amount)**

## 3.1 Data Transformation

Transformation of the variables:

```
Only LOG have output
Only some modelling algorithms improved with the transformed variables through log.
```
"""

#apply transformation to 'Cost_of_vehicle', 'Min_coverage' & 'Max_coverage'
trans1 = train_df['Max_coverage'].transform([np.log, np.exp, np.reciprocal, np.sqrt])

#Seeing which transformation is the best (ONLY *LOG* has output)
trans1.hist(bins=20)
plt.suptitle('Transformed Output')
plt.show()

#change age to the tranformation you have choosen
train_df['Cost_of_vehicle'] = train_df['Cost_of_vehicle'].transform([np.log])
train_df['Min_coverage'] = train_df['Min_coverage'].transform([np.log])
train_df['Max_coverage'] = train_df['Max_coverage'].transform([np.log])

"""Creating dummy variables:"""

# Extract the year from the date (cause it's makes more sense to just take the year)
train_df['Expiry_date'] = pd.to_datetime(train_df['Expiry_date'])
train_df['Expiry_year'] = pd.DatetimeIndex(train_df['Expiry_date']).year
train_df['Year_difference'] = train_df['Expiry_year'] - 2021 
train_df.head(5)

# Convert the amount into binary using the median as a break (less than 4048 = 0, more than 4048 = 1)
train_df["Binary_Amount"] = (train_df["Amount"] >= train_df["Amount"].median()).astype(int)
train_df.head(5)

# Split the data into X & y
X = new_train_df_categorical.drop('Binary_Amount', axis = 1).values
y = new_train_df_categorical['Binary_Amount']

y = y.astype(int)

print(X.shape)
print(y.shape)
train_df_categorical.head()

# Drop unused variables 
train_df_categorical = train_df.drop(columns=['Image_path', 'Expiry_date', 'Expiry_year','Amount'])
train_df_categorical.head(5)

# Convert "Insurance_company" to numeric representation (binary) 
cat_vars = ['Insurance_company']
for var in cat_vars:
    cat_list = 'var'+'_'+var
    cat_list = pd.get_dummies(train_df_categorical[var], prefix=var)

    train_df_categorical1 = train_df_categorical.join(cat_list)
    train_df_categorical = train_df_categorical1
    

data_vars = train_df_categorical.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]

new_train_df_categorical =train_df_categorical[to_keep]
new_train_df_categorical.columns.values
new_train_df_categorical = new_train_df_categorical.astype(int) #change all binary to int type
print(new_train_df_categorical.info())

# Split the data into X & y
X = new_train_df_categorical.drop('Binary_Amount', axis = 1).values
y = new_train_df_categorical['Binary_Amount']

y = y.astype(int)

print(X.shape)
print(y.shape)
train_df_categorical.head()

"""## 3.2 Feature selection """

# Run a Tree-based estimators (i.e. decision trees & random forests)
#dt = RandomForestClassifier(random_state=1, criterion = 'entropy')#, max_depth = 10)

dt = DecisionTreeClassifier(random_state=1, criterion = 'entropy')#, max_depth = 10)
dt.fit(X,y)

# Running Feature Importance
fi_col = []
fi = []

for i,column in enumerate(new_train_df_categorical.drop('Binary_Amount', axis = 1)):
    print('The feature importance for {} is : {}'.format(column, dt.feature_importances_[i]))
    
    fi_col.append(column)
    fi.append(dt.feature_importances_[i])

#RFE process 
X = new_train_df_categorical.loc[:, new_train_df_categorical.columns != 'Binary_Amount']
y = new_train_df_categorical['Binary_Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sel = RFE(DecisionTreeClassifier(),n_features_to_select = 10)
sel.fit(X_train, y_train)
sel.get_support()
features = X_train.columns[sel.get_support()]
print(features)

X = new_train_df_categorical[features].values
y = new_train_df_categorical['Binary_Amount']
print(X.shape)

"""# 4. Modelling

```
The model accuracy is much better where the highest result is 58.24% (KNN)
```

"""

# LOGISTIC REGRESSION
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
log_reg = LogisticRegression(random_state=10, solver = 'lbfgs')
log_reg.fit(X_train, y_train)
log_reg.predict(X_train)
y_pred = log_reg.predict(X_train)
pred_proba = log_reg.predict_proba(X_train)
print("The Testing Accuracy is: ", log_reg.score(X_test, y_test))

# SVM (kernel='rbf')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
svm_rbf.predict(X_train)
y_pred = svm_rbf.predict(X_train)
print("The Testing Accuracy is: ", svm_rbf.score(X_test, y_test))

# Decision tree
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,test_size=0.2)
dt =  DecisionTreeClassifier()
dt =  dt.fit(X_train,y_train)
dt_y_pred = dt.predict(X_test)
print('Accuracy:',metrics.accuracy_score(y_test,dt_y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, dt_y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, dt_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, dt_y_pred)))

# Gausian Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.predict(X_train)
y_pred = gnb.predict(X_train)
print("The Testing Accuracy is: ", gnb.score(X_test, y_test))

# K nearest neighbours 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# K is user specified , how do we know which K is the best to use ? 
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);

# K NEAREST NEIGHBOURS

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.2)
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
print("The Testing Accuracy is: ",knn.score(X_test, y_test))