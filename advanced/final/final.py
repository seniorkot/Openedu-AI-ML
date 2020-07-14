import numpy as np
import pandas as pd
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def print_results(variance_level: float, random_state: int,
                  rf_class: int, lgr_class: int, dt_class: int, files: list):
    (X_train, Y_train), (x_pred, y_pred) = mnist.load_data()

    dim = 28*28
    X_train = X_train.reshape(len(X_train), 28*28)

    pca_m = PCA(n_components=dim, svd_solver='full')
    _ = pca_m.fit_transform(X_train)
    explained_variance_2 = \
        np.round(np.cumsum(pca_m.explained_variance_ratio_), 3)

    res = -1
    for i in range(dim):
        if explained_variance_2[i] > variance_level:
            res = i + 1
            break

    print('Coords minimum for ' + str(variance_level) + ': ' + str(res))

    pca = PCA(n_components=res, svd_solver='full')
    X_transf = pca.fit_transform(X_train)
    x_train, x_test, y_train, y_test = train_test_split(X_transf, Y_train,
                                                        test_size=0.3,
                                                        random_state=
                                                        random_state)

    print('Average of 0 col: ' + str(np.average(x_train[:, 0])))

    # Random Forrest
    forrest = RandomForestClassifier(criterion='gini', min_samples_leaf=10,
                                     max_depth=20, n_estimators=10,
                                     random_state=random_state)
    clf_forrest = OneVsRestClassifier(forrest).fit(x_train, y_train)
    y_pred = clf_forrest.predict(x_test)
    print('Random forrest for class ' + str(rf_class) + ' true: ' +
          str(confusion_matrix(y_test, y_pred)[rf_class][rf_class]))

    # Logistic Regression
    lgr = LogisticRegression(solver='lbfgs', random_state=random_state)
    clf_lgr = OneVsRestClassifier(lgr).fit(x_train, y_train)
    y_pred = clf_lgr.predict(x_test)
    print('Logistic regression for class ' + str(lgr_class) + ' true: ' +
          str(confusion_matrix(y_test, y_pred)[lgr_class][lgr_class]))

    # Decision Tree
    tree = DecisionTreeClassifier(criterion='gini', min_samples_leaf=10,
                                  max_depth=20, random_state=random_state)
    clf_tree = OneVsRestClassifier(tree).fit(x_train, y_train)
    y_pred = clf_tree.predict(x_test)
    print('Decision tree for class ' + str(dt_class) + ' true: ' +
          str(confusion_matrix(y_test, y_pred)[dt_class][dt_class]))

    # Predict files
    data = pd.read_csv('final/data/pred_for_task.csv', encoding='utf-8',
                       index_col='FileName')

    # File 0
    file = data.loc[files[0]]
    label = file.Label
    file_rsh = file.drop('Label').to_numpy().reshape(1, -1)
    file_transf = pca.transform(file_rsh)
    pred = clf_forrest.predict_proba(file_transf)
    print('File 0 proba: ' + str(pred[:, label]))
    print(pred)

    # File 0
    file = data.loc[files[1]]
    label = file.Label
    file_rsh = file.drop('Label').to_numpy().reshape(1, -1)
    file_transf = pca.transform(file_rsh)
    pred = clf_lgr.predict_proba(file_transf)
    print('File 1 proba: ' + str(pred[:, label]))

    # File 0
    file = data.loc[files[2]]
    label = file.Label
    file_rsh = file.drop('Label').to_numpy().reshape(1, -1)
    file_transf = pca.transform(file_rsh)
    pred = clf_tree.predict_proba(file_transf)
    print('File 2 proba: ' + str(pred[:, label]))
