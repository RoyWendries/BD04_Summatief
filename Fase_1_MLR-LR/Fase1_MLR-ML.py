import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


class MLModel:
    def __init__(self, model, modelName):
        self.model = model
        self.modelName = modelName

    # Read Data
    def readData(self):
        xls = "ToyotaCorolla.xlsx"
        self.df = pd.read_excel(xls, "data")

    # Define X and y
    def df_Setter(self, X,  y):
        self.X = self.df[X]
        self.y = self.df[y]

    # Split data into trainings data and test data
    def split_Data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=0.6)

    # Set data to make predicions on
    def predict_Data(self, data1, data2, data3):
        self.Xpredict1 = data1
        self.Xpredict2 = data2
        self.Xpredict3 = data3

    # Fit data to model
    def Regfitter(self):
        self.data = self.model()
        self.data.fit(self.X_train.values, self.y_train.values)

    def results(self, predict, st, nd):
        self.Regfitter()
        print('\nScore of', self.modelName, 'on trainings data: ',
              self.data.score(self.X_train.values, self.y_train.values))
        print('Score of', self.modelName, 'on testing data: ',
              self.data.score(self.X_test.values, self.y_test.values))

        # Make predictions based on the predict data
        predictions = [self.Xpredict1, self.Xpredict2, self.Xpredict3]
        for e in predictions:
            print('Predicted', predict, 'of a car on the basis of a/an', st,
                  'of', e[0], 'and a/an', nd, 'of', e[1])


# Calling MLR
mlr = MLModel(LinearRegression, 'MLR')
mlr.readData()
mlr.df_Setter(['Age_08_04', 'KM'], 'Price')
mlr.split_Data()
mlr.predict_Data([5, 100], [15, 100000], [20, 30000])
mlr.results('cost', 'age', 'distance traveled')

# Calling ML
ml = MLModel(LogisticRegression, 'ML')
ml.readData()
ml.df_Setter(['Mfg_Year', 'Price'], 'Fuel_Type')
ml.split_Data()
ml.predict_Data([1998, 9000], [1999, 8671], [2002, 13500])
ml.results('fuel type', 'build year', 'cost')
