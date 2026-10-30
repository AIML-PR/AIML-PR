# AIML-PRAC
```exp1
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

```exp2
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
```exp3
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
