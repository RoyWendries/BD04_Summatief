from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import svm
from BD04_Summatief_Class import MLModel


# Calling MLPC with 2 features
mlpc = MLModel(MLPClassifier, 'MLPC', ['build year', 'cost'], NN=True)
mlpc.df_Setter(['Mfg_Year', 'Price'], 'Fuel_Type')
mlpc.predict_Data([1998, 9000], [1999, 8671], [2002, 13500])
mlpc.results('fuel type', layers=3)
mlpc.MLP_plotter()

# Calling MLPC with 3 features
mlpc2 = MLModel(MLPClassifier, 'MLPC', [
    'build year', 'cost', 'quarterly tax'], NN=True)
mlpc2.df_Setter(['Mfg_Year', 'Price', 'Quarterly_Tax'], 'Fuel_Type')
mlpc2.predict_Data([1998, 9000, 100], [1999, 8671, 210], [2002, 13500, 50])
mlpc2.results('fuel type', layers=5)
mlpc2.MLP_plotter()

# Calling MLPR with 2 features and 3 nodes in the hidden layer
mlpr = MLModel(MLPRegressor, 'MLPR', [
    'age', 'distance traveled'], NN=True)
mlpr.df_Setter(['Age_08_04', 'KM'], 'Price')
mlpr.predict_Data([5, 100], [15, 100000], [20, 30000])
mlpr.results('cost', layers=3)
mlpr.MLP_plotter()

# Calling MLPR with 3 features and 4 nodes in the hidden layer
mlpr2 = MLModel(MLPRegressor, 'MLPR', [
    'age', 'distance traveled', 'quarterly tax'], NN=True)
mlpr2.df_Setter(['Age_08_04', 'KM', 'Quarterly_Tax'], 'Price')
mlpr2.predict_Data([5, 100, 100], [15, 100000, 210], [20, 30000, 50])
mlpr2.results('cost', layers=5)
mlpr2.MLP_plotter()

# Calling SVC with 2 features
svmc = MLModel(svm.SVC, 'SVMC', ['build year', 'cost'])
svmc.df_Setter(['Mfg_Year', 'Price'], 'Fuel_Type')
svmc.predict_Data([1998, 9000], [1999, 8671], [2002, 13500])
svmc.results('fuel type')
svmc.SVM_plotter()

# Calling SVC with 3 features
svmc2 = MLModel(svm.SVC, 'SVMC', [
    'build year', 'cost', 'quarterly tax'])
svmc2.df_Setter(['Mfg_Year', 'Price', 'Quarterly_Tax'], 'Fuel_Type')
svmc2.predict_Data([1998, 9000, 100], [1999, 8671, 210], [2002, 13500, 50])
svmc2.results('fuel type')

# Calling SVR with 2 features
svmr = MLModel(svm.SVR, 'SVMR', [
    'age', 'distance traveled'])
svmr.df_Setter(['Age_08_04', 'KM'], 'Price')
svmr.predict_Data([5, 100], [15, 100000], [20, 30000])
svmr.results('cost')
svmr.SVM_plotter(predictor='Price')

# Calling SVR with 3 features
svmr2 = MLModel(svm.SVR, 'SVMR', [
    'age', 'distance traveled', 'quarterly tax'])
svmr2.df_Setter(['Age_08_04', 'KM', 'Quarterly_Tax'], 'Price')
svmr2.predict_Data([5, 100, 100], [15, 100000, 210], [20, 30000, 50])
svmr2.results('cost')
