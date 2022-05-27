import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

# Load data
xls = "ToyotaCorolla.xlsx"
df = pd.read_excel(xls, "data")

# Define X and y
X_MLR = df[['Age_08_04', 'KM']]
y_MLR = df['Price']

X_LR = df[['Mfg_Year', 'Price']]
y_LR = df['Fuel_Type']

# Split data into trainings data and test data
X_MLR_train, X_MLR_test, y_MLR_train, y_MLR_test = train_test_split(
    X_MLR, y_MLR, train_size=0.6)

X_LR_train, X_LR_test, y_LR_train, y_LR_test = train_test_split(
    X_LR, y_LR, train_size=0.6)

# New numbers,  Age and KM to predict cost
X_MLRnew1 = [[5, 100]]
X_MLRnew2 = [[15, 100000]]
X_MLRnew3 = [[20, 30000]]


# New numbers, Year and Price to predict fuel type
X_LRnew1 = [[1998, 9000]]
X_LRnew2 = [[1999, 8671]]
X_LRnew3 = [[2002, 13500]]

# Function to fit data with regression technieques


def Regfitter(X, y, model):
    if model == 'MLR':
        data = LinearRegression()
    elif model == 'LR':
        data = LogisticRegression(max_iter=1000)
    data.fit(X, y)
    return data


# Print score and predictions
models = ['MLR', 'LR']
for i in models:
    if i == 'MLR':
        X_train = X_MLR_train
        y_train = y_MLR_train
        X_test = X_MLR_test
        y_test = y_MLR_test
        Xnew1 = X_MLRnew1
        Xnew2 = X_MLRnew2
        Xnew3 = X_MLRnew3
    elif i == 'LR':
        X_train = X_LR_train
        y_train = y_LR_train
        X_test = X_LR_test
        y_test = y_LR_test
        Xnew1 = X_LRnew1
        Xnew2 = X_LRnew2
        Xnew3 = X_LRnew3

    model = Regfitter(X_train.values, y_train.values, i)
    if i == 'MLR':
        print('R2 Score of', i, 'on trainings data: ',
              model.score(X_train.values, y_train.values))
        print('R2 Score of', i, 'on testing data: ',
              model.score(X_test.values, y_test.values))
        print('Predicted cost of the car at 5 years old and 100 KM: ',
              model.predict(Xnew1))
        print('Predicted cost of the car at 15 years old and 100,000 KM: ',
              model.predict(Xnew2))
        print('Predicted cost of the car at 20 years old and 30,000 KM: ',
              model.predict(Xnew3), "\n")

    elif i == 'LR':
        print('Mean accuracy score of', i, 'on trainings data: ',
              model.score(X_train.values, y_train.values))
        print('Mean accuracy score of', i, 'on testing data: ',
              model.score(X_test.values, y_test.values))
        print('Predicted fuel type of a car built in 1998 and costs 9000: ',
              model.predict(Xnew1))
        print('Predicted fuel type of a car built in 1999 and costs 8671:',
              model.predict(Xnew2))
        print('Predicted fuel type of a car built in 2002 and costs 13500:',
              model.predict(Xnew3), "\n")
