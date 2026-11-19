# Loading the libraries
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error


# Loading dataset
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
dataset = read_csv('housing.csv', header=None, names=names, usecols=range(14))

#shape
print(dataset.shape)

#types
print(dataset.dtypes)

#head
print(dataset.head(20))

#descriptions
set_option('display.precision', 1)
print(dataset.describe())

#correlation
set_option('display.precision',2)
print(dataset.corr(method='pearson', numeric_only=True))

#Data visualisation
#histograms
dataset.hist (sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()

#density plot
dataset.plot(kind='density', subplots=True, layout=(4,4),sharex=False, sharey=False, fontsize=1)
pyplot.show()

#box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=8)
pyplot.show()


#scatter plot matrix
scatter_matrix(dataset)
pyplot.show()


#correlation matrix
fig = pyplot.figure()
ax= fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = numpy.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
test_size=validation_size, random_state=seed)
# Test options and evaluation metric
num_folds = 10
scoring = 'neg_mean_squared_error'

def compare(models, title):
    results, model_names = [], []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        model_names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    fig = pyplot.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(model_names)
    pyplot.show()

models = [('LR', LinearRegression()), ('LASSO', Lasso()), ('EN', ElasticNet()),
          ('KNN', KNeighborsRegressor()), ('CART', DecisionTreeRegressor()),
          ('SVR', SVR())]
compare(models, 'Algorithm Comparison')