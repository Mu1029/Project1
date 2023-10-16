
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

#plt.figure()


#  Step 3

    # train/test 80/20 data split and randomization

X_matrix = df[['X', 'Y', 'Z']]
print(X_matrix)

y = df['Step']

X_matrix_train, X_matrix_test, y_train, y_test = train_test_split(X_matrix, y, test_size=0.2, random_state=30)

    # creating full train/test set matrices

full_train_matrix = pd.DataFrame({'X': X_matrix_train['X'], 'Y': X_matrix_train['Y'], 'Z': X_matrix_train['Z'], 'Step': y_train})
full_test_matrix = pd.DataFrame({'X': X_matrix_test['X'], 'Y': X_matrix_test['Y'], 'Z': X_matrix_test['Z'], 'Step': y_test})

print(full_train_matrix)

    # creating correlation matrix

correlation_matrix = full_train_matrix.corr(method='spearman')

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()

    # dropping collinear variable (X)
  
collinear_var = ['X']

full_train_matrix = full_train_matrix.drop(columns=['X'])
full_test_matrix = full_test_matrix.drop(columns=['X'])
X_matrix_train = X_matrix_train.drop(columns=['X'])

print(full_train_matrix)
print(full_test_matrix)


#  Step 4 

    # model training (3 models)
  
model_1 = LinearRegression()
model_1.fit(X_matrix_train, y_train)

param_grid_lr = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'positive': [True, False],
}

grid_search_lr = GridSearchCV(model_1, param_grid_lr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_lr.fit(X_matrix_train, y_train)

best_hyperparams_lr = grid_search_lr.best_params_
best_model_1 = grid_search_lr.best_estimator_

print("\nBest Hyperparameters (LR): \n", best_hyperparams_lr)
print("Best Model (LR): \n", best_model_1)


model_2 = DecisionTreeRegressor(random_state=30)
model_2.fit(X_matrix_train, y_train)

param_grid_dt = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}

grid_search_dt = GridSearchCV(model_2, param_grid_dt, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_dt.fit(X_matrix_train, y_train)

best_hyperparams_dt = grid_search_dt.best_params_
best_model_2 = grid_search_dt.best_estimator_

print("\nBest Hyperparameters (DT): \n", best_hyperparams_dt)
print("Best Model (DT): \n", best_model_2)

model_3 = RandomForestRegressor(random_state=30)
model_3.fit(X_matrix_train, y_train)

param_grid_rf = {
   'n_estimators': [100, 200, 300],
   'max_depth': [None, 10, 20, 30],
   'min_samples_split': [2, 5, 10],
   'min_samples_leaf': [1, 2, 4],
   'max_features': ['sqrt', 'log2']
}

grid_search_rf = GridSearchCV(model_3, param_grid_rf, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_rf.fit(X_matrix_train, y_train)

best_hyperparams_rf = grid_search_rf.best_params_
best_model_3 = grid_search_rf.best_estimator_

print("\nBest Hyperparameters (RF): ", best_hyperparams_rf)
print("Best Model (RF): \n", best_model_3)


#  Step 5

    # Metrics
    
y_pred1 = model_1.predict(X_matrix_test)
y_pred2 = model_2.predict(X_matrix_test)
y_pred3 = model_3.predict(X_matrix_test)


mae1 = mean_absolute_error(y_test, y_pred1)
print("\nMean Absolute Error (Model 1): ", mae1)

mae2 = mean_absolute_error(y_test, y_pred2)
print("Mean Absolute Error (Model 2): ", mae2)

mae3 = mean_absolute_error(y_test, y_pred3)
print("Mean Absolute Error (Model 3): ", mae3)


mse1 = mean_squared_error(y_test, y_pred1)
print("\nMean Square Error (Model 1): ", mse1)

mse2 = mean_squared_error(y_test, y_pred2)
print("Mean Square Error (Model 2): ", mse2)

mse3 = mean_squared_error(y_test, y_pred3)
print("Mean Square Error (Model 3): ", mse3)

