from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from BD04_Summatief_Class import MLModel

# Calling DT classifier with 2 features
dtc = MLModel(tree.DecisionTreeClassifier, 'DTC', ['build year', 'cost'])
dtc.df_Setter(['Mfg_Year', 'Price'], 'Fuel_Type')
dtc.predict_Data([1998, 9000], [1999, 8671], [2002, 13500])
dtc.results('fuel type')
dtc.DT_plotter()

# Calling DT classifier with 3 features
dtc2 = MLModel(tree.DecisionTreeClassifier, 'DTC', [
               'build year', 'cost', 'quarterly tax'])
dtc2.df_Setter(['Mfg_Year', 'Price', 'Quarterly_Tax'], 'Fuel_Type')
dtc2.predict_Data([1998, 9000, 100], [1999, 8671, 210], [2002, 13500, 50])
dtc2.results('fuel type')
dtc2.DT_plotter()


# Calling DT regressor with 2 features (max_depth=4 to stop overfitting and
# maintain consistent accuracy between train and test data)
dtr = MLModel(tree.DecisionTreeRegressor, 'DTR', [
    'age', 'horse power'], MaxDepth=True)
dtr.df_Setter(['Age_08_04', 'HP'], 'Price')
dtr.predict_Data([5, 100], [30, 180], [40, 69])
dtr.results('cost')
dtr.DT_plotter()

# Calling DT regressor with 3 features (max_depth=4 to stop overfitting and
# maintain consistent accuracy between train and test data)
dtr2 = MLModel(tree.DecisionTreeRegressor, 'DTR', [
    'age', 'horse power', 'quarterly tax'], MaxDepth=True)
dtr2.df_Setter(['Age_08_04', 'HP', 'Quarterly_Tax'], 'Price')
dtr2.predict_Data([5, 100, 100], [30, 180, 210], [40, 69, 50])
dtr2.results('cost')
dtr2.DT_plotter()


# Calling RF classifier with 2 features (max_depth=4 to stop overfitting and
# maintain consistent accuracy between train and test data)
rfc = MLModel(RandomForestClassifier, 'RFC', [
              'build year', 'cost'], MaxDepth=True)
rfc.df_Setter(['Mfg_Year', 'Price'], 'Fuel_Type')
rfc.predict_Data([1998, 9000], [1999, 8671], [2002, 13500])
rfc.results('fuel type')
rfc.RF_plotter()

# Calling RF classifier with 3 features (max_depth=4 to stop overfitting and
# maintain consistent accuracy between train and test data)
rfc2 = MLModel(RandomForestClassifier, 'RFC', [
               'build year', 'cost', 'quarterly tax'], MaxDepth=True)
rfc2.df_Setter(['Mfg_Year', 'Price', 'Quarterly_Tax'], 'Fuel_Type')
rfc2.predict_Data([1998, 9000, 100], [1999, 8671, 210], [2002, 13500, 50])
rfc2.results('fuel type')
rfc2.RF_plotter()

# Calling RF classifier with 11 features
rfc2 = MLModel(RandomForestClassifier, 'RFC_Full', ['manifacture year', 'cost', 'quarterly tax', 'manifacture month',
                                                    'horse power', 'automatic', 'cc', 'doors', 'cylinders', 'gears', 'weight'], MaxDepth=1)
rfc2.df_Setter(['Mfg_Year', 'Price', 'Quarterly_Tax', 'Mfg_Month', 'HP',
               'Automatic', 'CC', 'Doors', 'Cylinders', 'Gears', 'Weight'], 'Fuel_Type')
rfc2.results('fuel type')
rfc2.RF_plotter()


# Calling RF regressor with 2 features
rfr = MLModel(RandomForestRegressor, 'RFR', [
    'age', 'horse power'])
rfr.df_Setter(['Age_08_04', 'HP'], 'Price')
rfr.predict_Data([5, 100], [30, 180], [40, 69])
rfr.results('cost')
rfr.RF_plotter()

# Calling RF regressor with 3 features
rfr2 = MLModel(RandomForestRegressor, 'RFR', [
    'age', 'horse power', 'quarterly tax'])
rfr2.df_Setter(['Age_08_04', 'HP', 'Quarterly_Tax'], 'Price')
rfr2.predict_Data([5, 100, 100], [30, 180, 210], [40, 69, 50])
rfr2.results('cost')
rfr2.RF_plotter()

# Calling RF regressor with 12 features
rfr2 = MLModel(RandomForestRegressor, 'RFR_Full', ['age', 'distance traveled', 'quarterly tax', 'manifacture month',
                                                   'manifacture year', 'horse power', 'automatic', 'cc', 'doors', 'cylinders', 'gears', 'weight'])
rfr2.df_Setter(['Age_08_04', 'KM', 'Quarterly_Tax', 'Mfg_Month', 'Mfg_Year',
               'HP', 'Automatic', 'CC', 'Doors', 'Cylinders', 'Gears', 'Weight'], 'Price')
rfr2.results('cost')
rfr.RF_plotter()
