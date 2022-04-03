import pandas as pd
import pickle

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

validation_data = pd.read_csv("validation.csv")

svm_model = pickle.load(open("modelSVM.pkl", "rb"))

X = validation_data.iloc[:, :2]
y = validation_data.iloc[:, 2]

svm_prediction = svm_model.predict(X)

print("Accuracy Score: ", accuracy_score(y, svm_prediction) * 100)
print("F1 Score: ", f1_score(y, svm_prediction))
print("Confusion Matrix: ", confusion_matrix(y, svm_prediction))
