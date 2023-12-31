import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from datetime import datetime

# optimal model
time1 = datetime.now()
col = ['Follicle No. (R)', 'Follicle No. (L)', 'Skin darkening (Y/N)', 'hair growth(Y/N)',
       'Weight gain(Y/N)', 'Cycle(R/I)', 'Fast food (Y/N)', 'Pimples(Y/N)']

# read datasets
data = pd.read_csv("PCOS.csv")
X = data[col]
y = data["PCOS (Y/N)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# creat model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# evolving model
print(model.score(X_test, y_test))
time2 = datetime.now()
first = time2 - time1
# Duration of the process
print(first)

# basic model
time1 = datetime.now()
# read datasets
data = pd.read_csv("PCOS.csv")
X = data.drop("PCOS (Y/N)", axis=1)
y = data["PCOS (Y/N)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# creat model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# evolving model
print(model.score(X_test, y_test))
time2 = datetime.now()
sec = time2 - time1
# Duration of the process
print(sec)
