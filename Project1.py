
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# dependant variable is the step #, independant variables are the coordinates (X,Y,Z)

#  Step 1

 # reading csv data file

data = 'Project 1 Data.csv'
df = pd.read_csv(data)


#  Step 2

 # data info and summary

print('\n', 'Data info: ', '\n')
print(df.info(), '\n')

print(df.head(),'\n')
print(df.tail(),'\n')

total = np.sum(df)
print(total, '\n')

print('Stat Summary: ', '\n')

print(df.describe())

 # data visualization plots

x = df['X']
y = df['Step']
plt.plot(x, y)
plt.grid()

plt.xlabel('X')
plt.ylabel('Step #')
plt.title('X vs Step')

x = df['Y']
y = df['Step']
plt.plot(x, y)
plt.grid()

plt.xlabel('Y')
plt.ylabel('Step #')
plt.title('Y vs Step')

x = df['Z']
y = df['Step']
plt.scatter(x, y)
plt.grid()

plt.xlabel('Z')
plt.ylabel('Step')
plt.title('Z vs Step (Scatter)')


plt.hist(df, bins=20)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.legend(['X', 'Y', 'Z', 'Step'])





























































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
