import sklearn as sk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures # for a) ii)
from sklearn.linear_model import LogisticRegression # for a) ii)
from sklearn import linear_model # for a) ii)
from sklearn.svm import LinearSVC # for b) i)

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


# a) ii) Train Logistic Regression classifier on data
# coded with help from GeeksforGeeks: https://www.geeksforgeeks.org/machine-learning/logistic-regression-with-polynomial-features/ (POLYNOMIAL)
xConcat = pd.concat([X1, X2], axis = "columns") # concatenate X1 and X2 to pass them into logistic regression function
logisticReg = linear_model.LogisticRegression() # set up logistic regression model object (https://realpython.com/pandas-merge-join-and-concat/)
logisticReg.fit(xConcat, Y) # train LR object by passing in concatenated X1, X2 and label Y

# polyLogReg = PolynomialFeatures(degree=2) # set up polynomial logistic regression to be quadratic
# XTrainPoly = polyLogReg.fit_transform(xConcat) # convert X1 and X2 data to be able to work w/ polynomial logistic regression

# polyLogRegModel = LogisticRegression(penalty = None) # train logistic regression on polynomial X1 and X2
# polyLogRegModel.fit(XTrainPoly, Y) # Fit training data w/ labels

# print("Intercept: ", polyLogRegModel.intercept_, "\nCoefficients: ", polyLogRegModel.coef_) # print intercept & coefficients

# CODE FOR LINEAR LOGISTIC REGRESSION (Initial Solution)
# coded with help from SciKit-Learn : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# logisticReg = linear_model.LogisticRegression() # set up logistic regression model object (https://realpython.com/pandas-merge-join-and-concat/)
# logisticReg.fit(xConcat, Y) # train LR object by passing in concatenated X1, X2 and label Y

# a) iii) Predict & plot target values,& plot decision boundary
yPrediction = logisticReg.predict(xConcat)


# FOR POLYNOMIAL, REVISIT
# yPrediction = polyLogRegModel.predict(XTrainPoly) # predictions made on polynomial training data

# def decisionBoundaryPlot(X, y, model, poly) :
#     xMin, xMax = X.iloc[:,0].min() - 0.1, X.iloc[:,0].max() # get max & min X1 points, & add padding to be displayed on scatter plot (- 0.1)
#     yMin, yMax = X.iloc[:,1].min() - 0.1, X.iloc[:,1].max() # get max & min X2 points, & add padding to be displayed on scatter plot (- 0.1)

#     xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.001),
#                          np.arange(yMin, yMax, 0.001)) # create a grid for points to be plotted on, (0.001 affects decision boundary, making it appear less blocky)
#     Z = model.predict(poly.transform(np.c_[xx.ravel(), yy.ravel()])) # helps model predict where a point will go, by looking at every block in the grid
#     Z = Z.reshape(xx.shape) # alligns grid to plot decision boundary
#     plt.contour(xx, yy, Z, levels = [0.5], alpha = 0.8) # plot decision boundary, where Y = 0.5 (constrained by levels parameter)
#     plt.rcParams['figure.constrained_layout.use'] = True # adjusting spacing so labels, titles & ledgends don't overlap
#     plt.scatter(xConcat.iloc[:,0], Y, c = "green", marker = "d", label = "Training Data") # plot training data
#     plt.scatter(xConcat.iloc[:,0], yPrediction, c = "magenta", marker = ".", label = "Prediction Data") # plot predicted data
#     # plt.scatter(x1Pos, x2Pos, c = "red", marker = "+", label = "+1") # plot rows where Y = 1 on a scatter plot
#     # plt.scatter(x1Neg, x2Neg, c = "blue", marker = "_", label = "-1") # plot rows where Y = -1 on a scatter plot


#     plt.title("Visualisation of +/-1") # plot title
#     plt.xlabel("Column 0 data") # X axis label
#     plt.ylabel("Column 1 data") # Y axis label
#     plt.legend(loc = "upper right") # adding legend to plot and forcing it to top-right corner
#     plt.show()

# decisionBoundaryPlot(xConcat, Y, polyLogRegModel, polyLogReg)

# compute decision boundary in terms of X2
intercept = logisticReg.intercept_
coeff = logisticReg.coef_[0] # to get coefficients out from coef_
x1Coeff = coeff[0] # get X1 coefficient (B1)
x2Coeff = coeff[1] # get X2 coefficient (B2)
xmin = X1.min() # get minimum value in X1
xmax = X1.max() # get maximum value in X1
print(xmin) # debugging
print(xmax) # degubbing

c = -(intercept / x2Coeff) # B0 / B2
mxMin = -((x1Coeff * xmin) / x2Coeff) # get the minimum point
mxMax = -((x1Coeff * xmax) / x2Coeff) # get the maximum point

x2Min = mxMin + c # add intercept to shift the point
x2Max = mxMax + c # add intercept to shift the point
x2Points = [x2Min, x2Max] # add X2 points to array
x1Points = [xmin, xmax] # add X1 points to array
print(x2Min, x2Max) # debugging

# b) i) Train linear SVM models w/ range of penalty parameters
# C = 0.01
svmModel1 = LinearSVC(C = 0.001).fit(xConcat, Y) # Use a penalty of 0.001, and pass in concatenated X1, X2 and Y labels
print("C = 0.001\nIntercept: ", svmModel1.intercept_, "\nCoefficient: ", svmModel1.coef_, "\n") # print intercept & coefficients
yModel1 = svmModel1.predict(xConcat) # predict target values w/ SVM1, C = 0.001 [for b) ii)]

s1x1Pos = []
s1x2Pos = []
s1x1Neg = []
s1x2Neg = []
i1 = 0
for i1 in range(len(yModel1)):
    if(yModel1[i1] == 1):
        # if value at position idx is 1
        s1x1Pos.append(X1.iloc[i1]) # add x1 coordinate to list of x1 coordinates where Y = 1
        s1x2Pos.append(X2.iloc[i1]) # add x2 coordinate to list of x2 coordinates where Y = 1
    else:
        s1x1Neg.append(X1.iloc[i1]) # add x1 coordinate to list of x1 coordinates where Y = -1
        s1x2Neg.append(X2.iloc[i1]) # add x2 coordinate to list of x2 coordinates where Y = -1

# C = 1
svmModel3 = LinearSVC(C = 1).fit(xConcat, Y) # Use a penalty of 1, and pass in concatenated X1, X2 and Y labels
print("C = 1\nIntercept: ", svmModel3.intercept_, "\nCoefficient: ", svmModel3.coef_, "\n") # print intercept & coefficients
yModel3 = svmModel3.predict(xConcat) # predict target values w/ SVM3, C = 1 [for b) ii)]

s3x1Pos = []
s3x2Pos = []
s3x1Neg = []
s3x2Neg = []
i3 = 0
for i3 in range(len(yModel3)):
    if(yModel3[i3] == 1):
        # if value at position idx is 1
        s3x1Pos.append(X1.iloc[i3]) # add x1 coordinate to list of x1 coordinates where Y = 1
        s3x2Pos.append(X2.iloc[i3]) # add x2 coordinate to list of x2 coordinates where Y = 1
    else:
        s3x1Neg.append(X1.iloc[i3]) # add x1 coordinate to list of x1 coordinates where Y = -1
        s3x2Neg.append(X2.iloc[i3]) # add x2 coordinate to list of x2 coordinates where Y = -1

# C = 100
svmModel5 = LinearSVC(C = 100).fit(xConcat, Y) # Use a penalty of 100, and pass in concatenated X1, X2 and Y labels
print("C = 100\nIntercept: ", svmModel5.intercept_, "\nCoefficient: ", svmModel5.coef_, "\n") # print intercept & coefficients
yModel5 = svmModel5.predict(xConcat) # predict target values w/ SVM5, C = 100 [for b) ii)]

s5x1Pos = []
s5x2Pos = []
s5x1Neg = []
s5x2Neg = []
i5 = 0
for i5 in range(len(yModel5)):
    if(yModel5[i5] == 1):
        # if value at position idx is 1
        s5x1Pos.append(X1.iloc[i5]) # add x1 coordinate to list of x1 coordinates where Y = 1
        s5x2Pos.append(X2.iloc[i5]) # add x2 coordinate to list of x2 coordinates where Y = 1
    else:
        s5x1Neg.append(X1.iloc[i5]) # add x1 coordinate to list of x1 coordinates where Y = -1
        s5x2Neg.append(X2.iloc[i5]) # add x2 coordinate to list of x2 coordinates where Y = -1

# b) ii) Predict & plot target values in training data, along w/ decision boundary
# predictions done in conjunction w/ b) i)

# create & plot decision boundaries
# coded with help from GeeksforGeeks: https://www.geeksforgeeks.org/machine-learning/visualizing-support-vector-machines-svm-using-python/
# plot using (intercept + X1 Coefficient * X1 / X2 coefficient)
xValues = np.linspace(X[:,0].min(), X[:,0].max(), 50)
y1 = - (svmModel1.intercept_ + (svmModel1.coef_[0][0] * xValues) / svmModel1.coef_[0][1])
y2 = - (svmModel3.intercept_ + (svmModel3.coef_[0][0] * xValues) / svmModel3.coef_[0][1])
y3 = - (svmModel5.intercept_ + (svmModel5.coef_[0][0] * xValues) / svmModel5.coef_[0][1])

# Plot for b) ii)
plt.plot(xValues, y1, color = "red", linestyle = "-", label = "C = 0.001")
plt.plot(xValues, y2, color = "green", linestyle = "--", label = "C = 1")
plt.plot(xValues, y3, color = "blue", linestyle = ":", linewidth = 10, label = "C = 100")
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

# everything that needs to be plotted, will be coded here:
plt.rcParams['figure.constrained_layout.use'] = True # adjusting spacing so labels, titles & ledgends don't overlap

# Comment out all question a) plots when running plots for question b)
# Plot for a) i)
# plt.scatter(x1Pos, x2Pos, c = "red", marker = "+", label = "+1") # plot rows where Y = 1 on a scatter plot
# plt.scatter(x1Neg, x2Neg, c = "blue", marker = "_", label = "-1") # plot rows where Y = -1 on a scatter plot


# # Plot for a) ii)
# plt.scatter(xConcat.iloc[:,0], Y, c = "green", marker = "d", label = "Training Data") # plot training data
# plt.scatter(xConcat.iloc[:,0], yPrediction, c = "magenta", marker = ".", label = "Prediction Data") # plot predicted data


# # # Line plot for a) iii)
# plt.plot(x1Points, x2Points, linestyle = "solid", color = "darkgreen") # plot the decision boundary

# # Additional Info for a) plots
# plt.title("Visualisation of +/-1") # plot title
# plt.xlabel("Column 0 data") # X axis label
# plt.ylabel("Column 1 data") # Y axis label
# plt.legend(loc = "upper right") # adding legend to plot and forcing it to top-right corner
# plt.show()

# Plot for b) i)
plt.scatter(s1x1Pos, s1x2Pos, c = "red", marker = "s", label = "SVM1 +1 Predictions C = 0.001") # plot positive predictions for SVM1
plt.scatter(s1x1Neg, s1x2Neg, c = "black", marker = "s", label = "SVM1 -1 Predictions C = 0.001") # plot negative predictions for SVM1

plt.scatter(s3x1Pos, s3x2Pos, c = "green", marker = ".", label = "SVM3 Predictions C = 1") # plot postitive predictions for SVM3
plt.scatter(s3x1Neg, s3x2Neg, c = "orange", marker = ".", label = "SVM3 -1 Predictions C = 1") # plot negative predictions for SVM3

plt.scatter(s5x1Pos, s5x2Pos, c = "yellow", marker = "|", label = "SVM5 Predictions C = 100") # plot positive predictions for SVM5
plt.scatter(s5x1Neg, s5x2Neg, c = "deepskyblue", marker = "|", label = "SVM5 -1 Predictions C = 100") # plot negative predictions for SVM5

# additional plot information
plt.title("Predictions Visualisation for SVM with differing C Penalties")
plt.xlabel("Column 0 data")
plt.ylabel("Column 1 data")
plt.legend(loc = "upper right", bbox_to_anchor =(1.2, 1.2))
plt.tight_layout()
plt.show()