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


# Calling MLPR with 2 features and 25 nodes in the hidden layer
mlpr = MLModel(MLPRegressor, 'MLPR', [
    'age', 'horse power'], NN=True)
mlpr.df_Setter(['Age_08_04', 'HP'], 'Price')
mlpr.predict_Data([5, 100], [30, 180], [40, 69])
mlpr.results('cost', layers=25)
mlpr.MLP_plotter()

# Calling MLPR with 3 features and 20 nodes in the hidden layer
mlpr2 = MLModel(MLPRegressor, 'MLPR', [
    'age', 'horse power', 'quarterly tax'], NN=True)
mlpr2.df_Setter(['Age_08_04', 'HP', 'Quarterly_Tax'], 'Price')
mlpr2.predict_Data([5, 100, 100], [30, 180, 210], [40, 69, 50])
mlpr2.results('cost', layers=20)


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


# Calling SVR with 2 features using the poly kernel
svmr = MLModel(svm.SVR, 'SVMR', [
    'age', 'horse power'], kernel='poly')
svmr.df_Setter(['Age_08_04', 'HP'], 'Price')
svmr.predict_Data([5, 100], [30, 180], [40, 69])
print('\nUsing poly kernal')
svmr.results('cost')
svmr.SVM_plotter(predictor='Price')

# Calling SVR with 2 features using the rbf kernel
svmr1 = MLModel(svm.SVR, 'SVMR', [
    'age', 'horse power'], kernel='rbf')
svmr1.df_Setter(['Age_08_04', 'HP'], 'Price')
svmr1.predict_Data([5, 100], [30, 180], [40, 69])
print('\nUsing rbf kernal')
svmr1.results('cost')

# Calling SVR with 2 features using the sigmoid kernel
svmr2 = MLModel(svm.SVR, 'SVMR', [
    'age', 'horse power'], kernel='sigmoid')
svmr2.df_Setter(['Age_08_04', 'HP'], 'Price')
svmr2.predict_Data([5, 100], [30, 180], [40, 69])
print('\nUsing sigmoid kernal')
svmr2.results('cost')

# Calling SVR with 3 features using the poly kernel
svmr10 = MLModel(svm.SVR, 'SVMR', [
    'age', 'horse power', 'quarterly tax'], kernel='poly')
svmr10.df_Setter(['Age_08_04', 'HP', 'Quarterly_Tax'], 'Price')
svmr10.predict_Data([5, 100, 100], [30, 180, 210], [40, 69, 50])
print('\nUsing poly kernal')
svmr10.results('cost')
