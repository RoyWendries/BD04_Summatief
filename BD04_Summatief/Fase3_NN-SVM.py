from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings
import sklearn.exceptions
warnings.filterwarnings(
    "ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class MLModel:
    def __init__(self, model, modelName):
        self.model = model
        self.modelName = modelName
        self.readData()

    # Read Data
    def readData(self):
        xls = "ToyotaCorolla.xlsx"
        self.df = pd.read_excel(xls, "data")

    # Define X and y
    def df_Setter(self, X,  y):
        self.X = self.df[X]
        self.y = self.df[y]
        self.split_Data()

    # Split data into trainings data and test data
    def split_Data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=0.6)

    # Set data to make predicions on
    def predict_Data(self, data1, data2, data3):
        self.Xpredict1 = [data1]
        self.Xpredict2 = [data2]
        self.Xpredict3 = [data3]

    # Fit data to model
    def Regfitter(self):
        self.data = self.model()
        self.data.fit(self.X_train.values, self.y_train.values)

    def results(self, predict, features):
        self.Regfitter()
        print('\nScore of', self.modelName, 'on trainings data: ',
              self.data.score(self.X_train.values, self.y_train.values))
        print('Score of', self.modelName, 'on testing data: ',
              self.data.score(self.X_test.values, self.y_test.values))

        # Make predictions based on the predict data
        predictions = [self.Xpredict1, self.Xpredict2, self.Xpredict3]
        for e in predictions:
            length = len(e[0])
            i = 2
            category = 'and a/an ' + str(features[1]) + ' of ' + str(e[0][1])
            while length != 2:
                length = length - 1
                category = category + ' and a/an ' + \
                    str(features[i]) + ' of ' + str(e[0][i])
                i = i + 1
            print('Predicted', predict, 'of a car on the basis of a/an', features[0],
                  'of', e[0][0], category, ': ', self.data.predict(e))


'''
# Calling MLR with 2 features
mlr = MLModel(LinearRegression, 'MLR')
mlr.df_Setter(['Age_08_04', 'KM'], 'Price')
mlr.predict_Data([5, 100], [15, 100000], [20, 30000])
mlr.results('cost', ['age', 'distance traveled'])
mlr.MLR_plotter('cost', ['age', 'distance traveled'])
'''
