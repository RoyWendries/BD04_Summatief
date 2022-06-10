from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
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

    def MLR_plotter(self, predict, features):
        X = self.X_test.values
        x = X[:, 0]
        y = X[:, 1]
        z = self.y_test.values
        r2 = self.data.score(X, z)

        x_pred = np.linspace(1, 80, 30)
        y_pred = np.linspace(1, 243000, 30)
        xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
        model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T
        predicted = self.data.predict(model_viz)

        plt.style.use('default')

        fig = plt.figure(figsize=(12, 4))

        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        axes = [ax1, ax2, ax3]

        for ax in axes:
            ax.plot(x, y, z, color='k', zorder=15,
                    linestyle='none', marker='o', alpha=0.5)
            ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted,
                       facecolor=(0, 0, 0, 0), s=20, edgecolor='#70b3f0')
            ax.set_xlabel(str(features[0]), fontsize=12)
            ax.set_ylabel(str(features[1]), fontsize=12)
            ax.set_zlabel(predict, fontsize=12)
            ax.locator_params(nbins=4, axis='x')
            ax.locator_params(nbins=5, axis='x')

        ax1.view_init(elev=35, azim=120)
        ax2.view_init(elev=4, azim=114)
        ax3.view_init(elev=60, azim=165)

        fig.suptitle('$R^2 = %.2f$' % r2, fontsize=20)

        fig.show()
        input('press a key to continue: ')

    def LR_plotter(self):
        print(classification_report(self.y_test,
              self.data.predict(self.X_test.values)))


# Calling MLR with 2 features
mlr = MLModel(LinearRegression, 'MLR')
mlr.df_Setter(['Age_08_04', 'KM'], 'Price')
mlr.predict_Data([5, 100], [15, 100000], [20, 30000])
mlr.results('cost', ['age', 'distance traveled'])
mlr.MLR_plotter('cost', ['age', 'distance traveled'])


# Calling MLR with 3 features
mlr2 = MLModel(LinearRegression, 'MLR')
mlr2.df_Setter(['Age_08_04', 'KM', 'Quarterly_Tax'], 'Price')
mlr2.predict_Data([5, 100, 100], [15, 100000, 210], [20, 30000, 50])
mlr2.results('cost', ['age', 'distance traveled', 'quarterly tax'])


# Calling LR with 2 features
lr = MLModel(LogisticRegression, 'LR')
lr.df_Setter(['Mfg_Year', 'Price'], 'Fuel_Type')
lr.predict_Data([1998, 9000], [1999, 8671], [2002, 13500])
lr.results('fuel type', ['build year', 'cost'])
lr.LR_plotter()

# Calling LR with 3 features
lr2 = MLModel(LogisticRegression, 'LR')
lr2.df_Setter(['Mfg_Year', 'Price', 'Quarterly_Tax'], 'Fuel_Type')
lr2.predict_Data([1998, 9000, 100], [1999, 8671, 210], [2002, 13500, 50])
lr2.results('fuel type', ['build year', 'cost', 'quarterly tax'])
lr2.LR_plotter()
