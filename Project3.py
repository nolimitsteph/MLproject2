#importing the libraries
import numpy
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

file = 'sonar.all-data'
dataset = read_csv(file, header= None)

#analyzing the data
# # shape
# print(dataset.shape)
#
# # types
# set_option('display.max_rows', 500)
# print(dataset.dtypes)
#
# # head
# set_option('display.width', 100)
# print(dataset.head(20))
#
# # descriptions, changing precision to 3 places
# set_option('display.precision', 3)
# print(dataset.describe())
#
# # class distribution
# print(dataset.groupby(60).size())

#histograms
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()

# density

dataset.plot(kind='density', subplots=True, layout=(8, 8), sharex=False,
             legend=False, fontsize=1)
pyplot.show()

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(8, 8), sharex=False,
             sharey=False, fontsize=1)
pyplot.show()




