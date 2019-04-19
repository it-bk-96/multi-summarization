import numpy as np
from joblib import dump
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from method.English.SVM.Utils import text_utils
from definitions import ROOT_DIR

X_train, Y_train, X_test, Y_test = text_utils.convert_features_svm(ROOT_DIR + '/method/English/SVM/Data/svm_features')

def train():
    model = SVC(kernel='rbf', C=32, gamma=0.01, probability=True) # probability=True
    model.fit(X_train, Y_train)
    dump(model, 'model')
    #Y_predict = model.predict_proba(X_test)
    Y_predict = model.predict(X_test)
    predictions = [round(value) for value in Y_predict]
    accuracy = accuracy_score(Y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


#
def train_test():
    for c in np.arange(1, 10, 2):
        c_end = c + 2
        C_range = np.logspace(c, c_end, 2)
        for g in np.arange(-5, 5, 2):
            g_end = g + 4
            gamma_range = np.logspace(g, g_end, 2)
            param_grid = dict(gamma=gamma_range, C=C_range)
            cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
            grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
            grid.fit(X_train, Y_train)

            print("The best param are %s with a score of %0.3f" % (grid.best_params_, grid.best_score_))
if __name__ == '__main__':
    train()
