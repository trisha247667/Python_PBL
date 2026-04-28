
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pickle

# Note: In a real workflow, you would import 'X' and 'df' from Phase 2
# For this script, we assume the variables are ready from the previous step.

def train_models(X, df):
    # 1. Prepare Target Variable
    y = df['label']

    # 2. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    # 4. Train SVM
    print("Training SVM...")
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    return lr_model, svm_model, X_test, y_test

print("Model Training logic defined.")
