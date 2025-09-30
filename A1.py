import sklearn as sk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#Data set:
# id:24--48--24 


# import data (week2.csv, given in assignment sheet)
df = pd.read_csv("week2.csv")
print(df.head())
X1 = df.iloc[:,0] # exctracting all rows from column 0
X2 = df.iloc[:,1] # extracting all rows from column 1
X = np.column_stack((X1, X2))
Y = df.iloc[:,2] # extracting all rows from column 2, want to plot this differenciating between +/-1

# a) i) Plot data & distinguish +/- 1

# set up arrays to store coordinates for X1 and X2 depending on the Y value (+/-1)
x1Pos = []
x1Neg = []
x2Pos = []
x2Neg = []

# index
idx = 0

# need to get contents of cell using .iloc, see w3schools for more info
for i in range(len(Y)):
    if(Y.iloc[idx] == 1):
        # if value at position idx is 1
        x1Pos.insert(idx, X1.iloc[idx]) # add x1 coordinate to list of x1 coordinates where Y = 1
        x2Pos.insert(idx, X2.iloc[idx]) # add x2 coordinate to list of x2 coordinates where Y = 1
        idx += 1 # increment index
    else:
        x1Neg.insert(idx, X1.iloc[idx]) # add x1 coordinate to list of x1 coordinates where Y = -1
        x2Neg.insert(idx, X2.iloc[idx]) # add x2 coordinate to list of x2 coordinates where Y = -1
        idx += 1 # increment index


plt.scatter(x1Pos, x2Pos, c = "red", marker = "+", label = "+1") # plot rows where Y = 1 on a scatter plot
plt.scatter(x1Neg, x2Neg, c = "blue", marker = "_", label = "-1") # plot rows where Y = -1 on a scatter plot
plt.title("Visualisation of +/-1") # plot title
plt.xlabel("Column 0 data") # X axis label
plt.ylabel("Column 1 data") # Y axis label
plt.legend(loc = "upper right") # adding legend to plot and forcing it to top-right corner
plt.show() # show plot

# a) ii) Train Logistic Regression classifier on data
# coded with help from SciKit-Learn : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

xConcat = pd.concat([X1, X2], axis = "columns") # concatenate X1 and X2 to pass them into logistic regression function
logisticReg = linear_model.LogisticRegression() # set up logistic regression model object (https://realpython.com/pandas-merge-join-and-concat/)
logisticReg.fit(xConcat, Y) # train LR object by passing in concatenated X1, X2 and label Y

def logRegProbability(logisticReg, xConcat):
    logRegOdds = logisticReg.coef_ * xConcat + logisticReg.intercept_
    odds = np.exp(logRegOdds)
    prob = odds / (1 + odds)
    print("Probability: ", prob)
    return prob

intercept = logisticReg.intercept_
coefficient = logisticReg.coef_
print("Intercept: ", intercept, "\nCoefficient: ", coefficient)