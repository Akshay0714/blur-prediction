import pandas as pd
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# generate score for the model


def generate_score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    confusion_mat = confusion_matrix(y_test, predictions)
    f1_scre = f1_score(y_test, predictions)
    return accuracy, confusion_mat, f1_scre


validation_data = pd.read_csv("validation.csv")

knn_model = pickle.load(open("knnModel.pkl", "rb"))

X = validation_data.iloc[:, :2]
y = validation_data.iloc[:, 2]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24)

# find the least error and store in variable
error1 = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred1 = knn.predict(X_test)
    error1.append(np.mean(y_test != y_pred1))

minIdx = error1.index(min(error1))

knn = KNeighborsClassifier(n_neighbors=minIdx + 1)
accuracy, confusion_mat, f1_scre = generate_score(
    knn, X_train, y_train, X_test, y_test)


print(f"Accuracy: {accuracy*100:0.2f} %")
print(f"F1 Score: {f1_scre:0.2f}")
