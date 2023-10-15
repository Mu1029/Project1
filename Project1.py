
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# dependant variable is the step #, independant variables are the coordinates (X,Y,Z)

# Question 1

data = 'Project 1 Data.csv'
df = pd.read_csv(data)

# Question 2

print('\n', df.info(), '\n')

print(df.head(),'\n')
print(df.tail(),'\n')


print('Stat Summary: ', '\n')

print(df.describe())


