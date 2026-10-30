# AIML-PRAC
```
Experiment 1: To understand the basic syntax of Python and use of key libraries such as NumPy, Pandas, and Matplotlib.
import numpy as np
# Create a 1D and a 2D array
vector = np.array([1, 2, 3, 4])
matrix = np.array([[1, 2], [3, 4]])
# Vectorized arithmetic (applied element-wise instantly)
squared_vector = vector ** 2
boosted_matrix = matrix + 10
#Common mathematical tools
mean_val = np.mean(vector)
# Computes mathematical mean
grid_space = np.arange(0, 10, 2) #Array from 0 to 8, stepping by 2

import pandas as pd
# Creating a DataFrame from a dictionary
raw_data = {
"Employee": ["Alice", "Bob", "Charlie", "David"],
"Department": ["HR", "IT", "IT", "Finance"],
"Salary": [50000, 75000, 82000, np.nan] # np.nan indicates a missing value
}
df = pd.DataFrame(raw_data)
# Exploring and cleaning data
print(df.head(2))
df_clean = df.dropna()
# Selecting & Filtering
it_staff = df[df["Department"] == "IT"] # Filter rows based on a condition
salaries = df["Salary"]
# Aggregation
avg_salary = df["Salary"].mean()

import matplotlib.pyplot as plt
# Sample coordinates
months = ["Jan", "Feb", "Mar", "Apr"]
sales = [1200, 1500, 1300, 1900]
#Build and customize a line plot
plt.plot(months, sales, marker ="o", color="blue", linestyle="--")
# Adding critical visual contexts
plt.title("Q1 Company Sales Performance")
plt.xlabel("Timeline")
plt.ylabel("Revenue ($)")
plt.grid(True)
# Render graph onto the desktop/notebook window
plt.show()
```

```
Experiment 2: To implement a linear regression model and evaluate its performance.
# Import Pandas for data processing
import pandas as pd
# Read the CSV file
dataset = pd.read_csv('C:/Users/khanm/Desktop/ML/Students.csv')
df = dataset.copy()
# Split into X (Independent) and Y (predicted)
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
# Create the Training and Test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (X, Y, test_size =0.3, random_state =1234)
# Train the Simple Linear Regression
from sklearn.linear_model import LinearRegression
std_reg = LinearRegression()
std_reg.fit(x_train, y_train)
# Predict the results
y_predict = std_reg.predict(x_test)
# Get the R-Squared
slr_score = std_reg.score(x_test, y_test)
print("R-Square = ",slr_score)
# Coefficient and Intercept
slr_coefficient = std_reg.coef_
slr_intercept = std_reg.intercept_
print("coefficint =", slr_coefficient)
print("Y_intercept = ",slr_intercept)
# Calculate the errors using RMSE
from sklearn.metrics import mean_squared_error
import math
slr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE error =",slr_rmse)
# plot the result using matplotlib
import matplotlib.pyplot as plt
plt.scatter(x_test, y_test)
plt.show()
plt.plot(x_test, y_predict)
plt.ylim(ymin=0)
plt.show()
```
```
Experiment 3: To implement a logistic regression model and evaluate its performance.
from sklearn import datasets
#Load Iris Dataset from Scikit learn Library.
iris=datasets.load_iris()
X=iris.data
y=iris.target
print(X.shape)
print(y.shape)
# Split the dataset into train-test split ratio.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size =0.3,random_state =123,stratify=y)
# Train Logistic Regression Model.
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_test)
from sklearn import metrics
print("\n Classification Reort \n")
print(metrics.classification_report(y_test,y_pred))
```
```
Experiment 4: To implement a Decision Tree classification model and evaluate its performance.
# Import libraries
import pandas as pd
# Read dataset
data = pd.read_csv('C:Desktop/IAIML-Lab/03_decisiontree_AdultIncome.csv')
#Check for Null values
data.isnull().sum(axis=0)
# Create Dummy variables
data.dtypes
data_prep = pd.get_dummies(data, drop_first=True)
# Create X and Y Variables
X = data_prep.iloc[:, :-1]
Y = data_prep.iloc[:, -1]
# Split the X and Y dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.3, random_state =1234, stratify =Y)
# Import and train classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state =1234)
dtc.fit(X_train, Y_train)
#Test the model
Y_predict = dtc.predict(X_test)
# Evaluate the model
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(Y_test, Y_predict)
score = dtc.score(X_test, Y_test)
print("Accuracy = ",score*100)
print(classification_report(Y_test,Y_predict))
print('Confusion matrix', cm)
```

```
Experiment 5: To implement K-Nearest Neighbour (KNN) classification model and evaluate its performance.
# importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
#importing datasets
data_set= pd.read_csv('C:/Users/khanm/Desktop/ML-python-codes/Dataset/suv_data.csv')
#Extracting Independent and dependent Variable
x = data_set.iloc[:, [2,3]].values
y = data_set.iloc[:, 4].values
# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.25, random_state =0)
#feature Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train_st= st_x.fit_transform(x_train)
x_test_st= st_x.transform(x_test)
#Fitting K-NN classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors =5, metric='minkowski', p=2)
classifier.fit(x_train_st, y_train)
#Predicting the test set result
y_pred= classifier.predict(x_test_st)
#Creating the Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm= confusion_matrix(y_test, y_pred)
print("Classification Report")
print(classification_report(y_test, y_pred))
```

```
Experiment 6: To implement Decision Tree and Random Forest classifications models and evaluate their performance.
# Import libraries
import pandas as pd
#Read dataset
data = pd.read_csv('C:/Users/ICTC/Desktop/IAIML-Lab/Dataset/03_decisiontree_AdultIncome.csv')
#Check for Null values
data.isnull().sum(axis=0)
#Create Dummy variables
data.dtypes
data_prep = pd.get_dummies(data, drop_first=True)
# Create X and Y Variables
X = data_prep.iloc[:, :-1]
Y = data_prep.iloc[:, -1]
# Split the X and Y dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.2, random_state =1234, stratify =Y)
# Import and train Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators =100,random_state =1234, bootstrap=True)
rfc.fit(X_train, Y_train)
# Test the RFC model
Y_predict = rfc.predict(X_test)
## Evaluate the model
from sklearn.metrics import confusion_matrix, classification_report
cm2 = confusion_matrix(Y_test, Y_predict)
print('confusion matrix for RF classifier: \n',cm2)
score = rfc.score(X_test, Y_test)
print('Random Forest classification report')
print(classification_report(Y_test, Y_predict))
```

```
Experiment 7: To implement Support Vector Machine (SVM) classification model and evaluate its performance.
from sklearn import datasets
iris=datasets.load_iris()
X = iris.data
y=iris.target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size =0.3, random_state =123,stratify=y)
from sklearn.svm import SVC
sv=SVC(kernel="linear", gamma =0.5, C=1.0)
sv.fit(X_train,y_train)
y_pred=sv.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print('% Accuracy ={:.2f}'.format(acc*100))
print('SVM classification report')
print(classification_report(y_test,y_pred))
```

```
Experiment 8: To implement K-Means Clustering classification model and evaluate its performance.   
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
#Step 2: Creating Custom Dataset
X,y = make_blobs(n_samples =500,n_features =2, centers =3,random_state =23)
#Step 3:Feature Scaling using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
#Step 4: Initializing Random Centroids
k=3
clusters ={}
np.random.seed(23)
for idx in range(k):
    center = 2*(2*np.random.random((X.shape[1],))-1)
    points = []
    cluster = {
        'center': center,
        'points': []
    }
    clusters[idx] = cluster
#Step 5: Defining Euclidean Distance
def distance(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))
#Step 6: Creating Assign and Update Functions
def assign_clusters (X, clusters):
    for idx in range(X.shape[0]):
        dist = []
        curr_x = X[idx]
        for i in range(k):
            dis = distance(curr_x,clusters[i]['center'])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters
def update_clusters(X, clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis =0)
            clusters[i]['center'] = new_center
            clusters[i]['points'] = []
    return clusters
#Step 7: Predicting the Cluster for the Data Points
def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance (X[i], clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred
#Step 8: Assigning, Updating and Predicting the Cluster Centers
max_iters =100
for _ in range(max_iters):
    old_centers = [clusters[i]['center'].copy() for i in range(k)]
    clusters = assign_clusters(X, clusters)
    clusters = update_clusters(X, clusters)
    new_centers = [clusters[i]['center'] for i in range(k)]
    if np.allclose(old_centers, new_centers):
        break
pred = pred_cluster(X, clusters)
#Step 9: Plotting Data Points with Predicted Cluster Centers
plt.scatter(X[:,0],X[:, 1], c=pred)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1], marker = '^', c='red')
plt.show()
```

```
Experiment 9: To implement the Principal Component Analysis (PCA) model for dimensionality reduction.   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
# construct a dataframe using pandas
df1=pd.DataFrame(data['data'],columns=data['feature_names'])
# Scale data before applying PCA
scaling = StandardScaler()
scaling.fit(df1)
Scaled_data = scaling.transform(df1)
# Set the n_components=3
principal=PCA(n_components=3)
principal.fit(Scaled_data)
x=principal.transform(Scaled_data)
print(principal.components_)
plt.figure(figsize=(10,10))
plt.scatter(x[:,0],x[:, 1],c=data['target'],cmap='plasma')
plt.xlabel('pc1')
plt.ylabel('pc2')
# import relevant libraries for 3d graph
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))
axis = fig.add_subplot(111, projection ='3d')
axis.scatter(x[:,0],x[:,1],x[:,2], c=data['target'],cmap='plasma')
axis.set_xlabel("PC1", fontsize =10)
axis.set_ylabel("PC2", fontsize =10)
axis.set_zlabel("PC3", fontsize =10)
print("\n Variance of Principal Components PC1, PC2, PC3 ")
print(principal.explained_variance_ratio_)
```

```
Experiment 10: To implement a single-layer perceptron and MLP using Python/Tensor Flow/Keras.
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#1. Generate synthetic linearly separable data for binary classification
X, y = make_classification(n_samples =1000, n_features =5, n_informative =3, n_redundant =2, random_state =42)
#Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state =42)
# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#2. Build the Single-Layer Perceptron model
slp_model = Sequential ([
    Dense(units =1, input_shape =(5,), activation='sigmoid')
])
#3. Compile the model
slp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("--- Single-Layer Perceptron Summary ---")
slp_model.summary()
#4. Train the model
print("\nTraining SLP...")
slp_model.fit(X_train, y_train, epochs =30, batch_size =32, verbose=1)
#5. Evaluate the model
loss, accuracy = slp_model.evaluate(X_test, y_test, verbose =0)
print(f"\nSLP Test Loss: {loss:.4f}")
print(f"SLP Test Accuracy: {accuracy: .4f}")
```
