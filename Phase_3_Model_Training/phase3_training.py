import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib

X = pd.read_csv("features.csv")
y = pd.read_csv("labels.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LogisticRegression()
svm = SVC()

lr.fit(X_train, y_train.values.ravel())
svm.fit(X_train, y_train.values.ravel())

joblib.dump(lr, "logistic_model.pkl")
joblib.dump(svm, "svm_model.pkl")

print("Phase 3 complete")
