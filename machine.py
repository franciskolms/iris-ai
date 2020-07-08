# msg = "new TNSz"
# x = 1
# x = 7

# print(float(x))

# one = 1
# two = 2
# sum = one + two
# print(sum)

# numbers = []
# strings = []
# names = ["John", "Eric", "Jessica"]

# numbers.append(1)
# numbers.append(2)
# numbers.append(3)
# strings.append("hello")
# strings.append("world")

# second_name = names[1]

# print(numbers)
# print(strings)
# print("The second name on the names list is %s" % second_name)

# remainder = 11 % 3
# print(remainder)

# lotsofhellos = "hey " * 30
# print(lotsofhellos)

# even_numbders = [2,4,6,8]
# odd_numbers = [1,3,5,7]
# all_numbers = odd_numbers + even_numbders
# print(all_numbers)

# print([1,3,5]*3)

# x = object()
# y = object()

# x_list = [x] * 10
# y_list = [y] * 10
# big_list = x_list + y_list

# print("x_list contains %d objects" % len(x_list))
# print("y_list contains %d objects" % len(y_list))
# print("big_list contains %d objects" % len(big_list))

# # testing code
# if x_list.count(x) == 10 and y_list.count(y) == 10:
#     print("Almost there...")
# if big_list.count(x) == 10 and big_list.count(y) == 10:
#     print("Great!")

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# shape
print(dataset.shape)

#head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())