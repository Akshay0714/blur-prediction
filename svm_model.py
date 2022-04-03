import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


def generate_score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    confusion_mat = confusion_matrix(y_test, predictions)
    f1_scre = f1_score(y_test, predictions)
    return accuracy, confusion_mat, f1_scre


train_data = pd.read_csv("train.csv")

X = train_data.iloc[:, :2]
y = train_data.iloc[:, 2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24)

print("Training Data: ", X_train.shape, y_train.shape)
print("Test Data: ", X_test.shape, y_test.shape)

svm_model = SVC(C=10, class_weight="balanced", gamma=0.1)
accuracy, confusion_mat, f1_scre = generate_score(
    svm_model, X_train, y_train, X_test, y_test)


print(f"Accuracy: {accuracy*100}%")
print(f"F1 Score: {f1_scre}")

name = 'modelSVM'

svm_model.fit(X, y)
pickle.dump(svm_model, open(name+".pkl", 'wb'))
