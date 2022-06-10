import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


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
        if self.modelName == 'DTR' or self.modelName == 'RFC' or self.modelName == 'RFC_Full':
            self.data = self.model(max_depth=4)
        else:
            self.data = self.model()
        self.data.fit(self.X_train.values, self.y_train.values)

    def results(self, predict, features):
        self.Regfitter()
        print('\nScore of', self.modelName, 'on trainings data: ',
              self.data.score(self.X_train.values, self.y_train.values))
        print('Score of', self.modelName, 'on testing data: ',
              self.data.score(self.X_test.values, self.y_test.values))

        if self.modelName != 'RFR_Full' and self.modelName != 'RFC_Full':
            # Make predictions based on the predict data
            predictions = [self.Xpredict1, self.Xpredict2, self.Xpredict3]
            for e in predictions:
                length = len(e[0])
                i = 2
                category = 'and a/an ' + \
                    str(features[1]) + ' of ' + str(e[0][1])
                while length != 2:
                    length = length - 1
                    category = category + ' and a/an ' + \
                        str(features[i]) + ' of ' + str(e[0][i])
                    i = i + 1
                print('Predicted', predict, 'of a car on the basis of a/an', features[0],
                      'of', e[0][0], category, ': ', self.data.predict(e))
        if self.modelName == 'DTC' or self.modelName == "DTR":
            self.DT_plotter()
        else:
            self.RF_plotter(features)

    def DT_plotter(self):
        tree.plot_tree(self.data, filled=True)
        plt.show()
        input('Press any key to continue: ')

    def RF_plotter(self, features):
        importances = self.data.feature_importances_
        indices = np.argsort(importances)

        plt.title('Feature Importances')
        plt.barh(range(len(indices)),
                 importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()
        input('Press any key to continue: ')


# Calling DT classifier with 2 features
dtc = MLModel(tree.DecisionTreeClassifier, 'DTC')
dtc.df_Setter(['Mfg_Year', 'Price'], 'Fuel_Type')
dtc.predict_Data([1998, 9000], [1999, 8671], [2002, 13500])
dtc.results('fuel type', ['build year', 'cost'])

# Calling DT classifier with 3 features
dtc2 = MLModel(tree.DecisionTreeClassifier, 'DTC')
dtc2.df_Setter(['Mfg_Year', 'Price', 'Quarterly_Tax'], 'Fuel_Type')
dtc2.predict_Data([1998, 9000, 100], [1999, 8671, 210], [2002, 13500, 50])
dtc2.results('fuel type', ['build year', 'cost', 'quarterly tax'])

# Calling DT regressor with 2 features (max_depth=4 to stop overfitting and
# maintain consistent accuracy between train and test data)
dtr = MLModel(tree.DecisionTreeRegressor, 'DTR')
dtr.df_Setter(['Age_08_04', 'KM'], 'Price')
dtr.predict_Data([5, 100], [15, 100000], [20, 30000])
dtr.results('cost', ['age', 'distance traveled'])

# Calling DT regressor with 3 features (max_depth=4 to stop overfitting and
# maintain consistent accuracy between train and test data)
dtr2 = MLModel(tree.DecisionTreeRegressor, 'DTR')
dtr2.df_Setter(['Age_08_04', 'KM', 'Quarterly_Tax'], 'Price')
dtr2.predict_Data([5, 100, 100], [15, 100000, 210], [20, 30000, 50])
dtr2.results('cost', ['age', 'distance traveled', 'quarterly tax'])

# Calling RF classifier with 2 features (max_depth=4 to stop overfitting and
# maintain consistent accuracy between train and test data)
rfc = MLModel(RandomForestClassifier, 'RFC')
rfc.df_Setter(['Mfg_Year', 'Price'], 'Fuel_Type')
rfc.predict_Data([1998, 9000], [1999, 8671], [2002, 13500])
rfc.results('fuel type', ['build year', 'cost'])

# Calling RF classifier with 3 features (max_depth=4 to stop overfitting and
# maintain consistent accuracy between train and test data)
rfc2 = MLModel(RandomForestClassifier, 'RFC')
rfc2.df_Setter(['Mfg_Year', 'Price', 'Quarterly_Tax'], 'Fuel_Type')
rfc2.predict_Data([1998, 9000, 100], [1999, 8671, 210], [2002, 13500, 50])
rfc2.results('fuel type', ['build year', 'cost', 'quarterly tax'])

# Calling RF classifier with 11 features
rfc2 = MLModel(RandomForestClassifier, 'RFC_Full')
rfc2.df_Setter(['Mfg_Year', 'Price', 'Quarterly_Tax', 'Mfg_Month', 'HP',
               'Automatic', 'CC', 'Doors', 'Cylinders', 'Gears', 'Weight'], 'Fuel_Type')
rfc2.results('fuel type', ['manifacture year', 'cost', 'quarterly tax', 'manifacture month',
             'horse power', 'automatic', 'cc', 'doors', 'cylinders', 'gears', 'weight'])

# Calling RF regressor with 2 features
rfr = MLModel(RandomForestRegressor, 'RFR')
rfr.df_Setter(['Age_08_04', 'KM'], 'Price')
rfr.predict_Data([5, 100], [15, 100000], [20, 30000])
rfr.results('cost', ['age', 'distance traveled'])

# Calling RF regressor with 3 features
rfr2 = MLModel(RandomForestRegressor, 'RFR')
rfr2.df_Setter(['Age_08_04', 'KM', 'Quarterly_Tax'], 'Price')
rfr2.predict_Data([5, 100, 100], [15, 100000, 210], [20, 30000, 50])
rfr2.results('cost', ['age', 'distance traveled', 'quarterly tax'])

# Calling RF regressor with 12 features
rfr2 = MLModel(RandomForestRegressor, 'RFR_Full')
rfr2.df_Setter(['Age_08_04', 'KM', 'Quarterly_Tax', 'Mfg_Month', 'Mfg_Year',
               'HP', 'Automatic', 'CC', 'Doors', 'Cylinders', 'Gears', 'Weight'], 'Price')
rfr2.results('cost', ['age', 'distance traveled', 'quarterly tax', 'manifacture month',
             'manifacture year', 'horse power', 'automatic', 'cc', 'doors', 'cylinders', 'gears', 'weight'])
