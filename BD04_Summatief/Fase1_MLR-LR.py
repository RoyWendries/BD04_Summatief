from BD04_Summatief_Class import MLModel
from sklearn.linear_model import LinearRegression, LogisticRegression


# Calling MLR with 2 features
mlr = MLModel(LinearRegression, 'MLR', [
    'age', 'horse power'])
mlr.df_Setter(['Age_08_04', 'HP'], 'Price')
mlr.predict_Data([5, 100], [30, 180], [40, 69])
mlr.results('cost')
mlr.MLR_plotter('cost')

# Calling MLR with 3 features
mlr2 = MLModel(LinearRegression, 'MLR', [
    'age', 'horse power', 'quarterly tax'])
mlr2.df_Setter(['Age_08_04', 'HP', 'Quarterly_Tax'], 'Price')
mlr2.predict_Data([5, 100, 100], [30, 180, 210], [40, 69, 50])
mlr2.results('cost')
input('Press any key to continue: ')


# Calling LR with 2 features
lr = MLModel(LogisticRegression, 'LR', ['build year', 'cost'])
lr.df_Setter(['Mfg_Year', 'Price'], 'Fuel_Type')
lr.predict_Data([1998, 9000], [1999, 8671], [2002, 13500])
lr.results('fuel type')
lr.LR_plotter()

# Calling LR with 3 features
lr2 = MLModel(LogisticRegression, 'LR', [
              'build year', 'cost', 'quarterly tax'])
lr2.df_Setter(['Mfg_Year', 'Price', 'Quarterly_Tax'], 'Fuel_Type')
lr2.predict_Data([1998, 9000, 100], [1999, 8671, 210], [2002, 13500, 50])
lr2.results('fuel type')
lr2.LR_plotter()
