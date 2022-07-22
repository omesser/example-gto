import pandas
import pathlib
import pickle

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

import mlem.api

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
test_size = 0.33
seed = 7

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

# Fit the model on training set
model = LogisticRegression(max_iter=300)
model.fit(X_train, Y_train)

# save the model to disk
filename = pathlib.Path(pathlib.Path.cwd() / "models" / "model_sklearn_lr.sav")
pickle.dump(model, open(filename, 'wb'))

# save this with mlem
mlem.api.save(model, str(pathlib.Path(pathlib.Path.cwd() / "models" / "model_sklearn_lr.sav")), external=False)
