from sklearn.svm import SVC
from joblib import dump
from scipy import sparse
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import text_utils

X_train, Y_train, X_test, Y_test = text_utils.convert_features_svm('/home/hieupd/PycharmProjects/multi_summari_svm/svm_features')

# X_train = sparse.load_npz("data/datatrainsvm1.npz")
# Y_train = text_utils.read_file_text("data/datatrainsvm_label1").split("\n")
# X_test = sparse.load_npz("data/datatestsvm1.npz").toarray()
# Y_test = text_utils.read_file_text("data/datatestsvm_label1").split("\n")
#
#
def train():
    model = SVC(kernel='rbf', C=100, gamma=0.01, probability=True)
    model.fit(X_train, Y_train)
    dump(model, 'model_test')
    #Y_predict = model.predict_proba(X_test)
    #print(Y_predict)
    #print(sorted(Y_predict, key= lambda x: x[0]))


#
def train_test():
    for c in np.arange(-2, 10, 2):
        c_end = c + 2
        C_range = np.logspace(c, c_end, 2)
        for g in np.arange(-9, 3, 2):
            g_end = g + 4
            gamma_range = np.logspace(g, g_end, 2)
            param_grid = dict(gamma=gamma_range, C=C_range)
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
            grid.fit(X_train, Y_train)

            print("The best param are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

train()