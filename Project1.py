
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D


# dependant variable is the step #, independant variables are the coordinates (X,Y,Z)

#  Step 1

 # reading csv data file

data = 'Project 1 Data.csv'
df = pd.read_csv(data)


#  Step 2

 # data info and summary

print('\n', 'Data info: ', '\n')
print(df.info(), '\n')

total = np.sum(df)
print('Value Sums: ', '\n')
print(total, '\n')

print('Stat Summary: ', '\n')

print(df.describe())

 # data visualization plots

x = df['X']
y = df['Step']
plt.scatter(x, y)
plt.grid()

plt.xlabel('X')
plt.ylabel('Step #')
plt.title('X vs Step (Scatter)')

x2 = df['Y']
y2 = df['Step']
plt.scatter(x2, y2)
plt.grid()

plt.xlabel('Y')
plt.ylabel('Step #')
plt.title('Y vs Step (Scatter)')

x3 = df['Z']
y3 = df['Step']
plt.scatter(x3, y3)
plt.grid()

plt.xlabel('Z')
plt.ylabel('Step')
plt.title('Z vs Step (Scatter)')


plt.hist(df, bins=20)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.legend(['X', 'Y', 'Z', 'Step'])

ax = plt.figure().add_subplot(111, projection='3d')
x4 = df['X']
y4 = df['Y']
z = df['Z']

ax.scatter(x4, y4, z, s=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D-coordinate Scatter Plot')

























































# Step 4

 # train/test 80/20 data split

# X_matrix = df[['X', 'Y', 'Z']]
# y = df['Step']

# X_matrix_train, X_matrix_test, y_train, y_test = train_test_split(X_matrix, y, test_size=0.2, random_state=30)

# print('\n', X_matrix_train,'\n')
# print(X_matrix_test,'\n')
# print(y_train,'\n')
# print(y_test)








# talk about what was found here ^
