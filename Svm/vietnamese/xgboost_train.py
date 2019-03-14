import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
import text_utils

# load data

X_train, y_train, X_test, y_test = text_utils.convert_features_svm('/home/hieupd/PycharmProjects/multi_summari_svm_vietnamese/svm_features')

model = XGBClassifier(learning_rate=0.02, n_estimators=200, booster= 'gbtree', max_delta_step=2, subsample=0.8)
model.fit(X_train, y_train)
dump(model, 'xgboost_model')
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
#print(y_pred)
# a = 0
# for i in range(len(y_test)):
#     if y_test[i] == predictions[i] and y_test[i] == 1:
#         a += 1
#
# print(len(y_test))
# print(a, a/len(y_test))
#
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


