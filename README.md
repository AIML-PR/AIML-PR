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


