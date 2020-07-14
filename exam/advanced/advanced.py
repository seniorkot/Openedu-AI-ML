import pandas as pd
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def print_results(rows_count: int, random_state: int, image_id: int,
                  princip_comps: int):
    (X_train, Y_train), (x_pred, y_pred) = mnist.load_data()
    data = pd.read_csv('advanced/data/DataForPrediction_FinalTask.csv',
                       encoding='utf-8', index_col='id')

    # Select first N rows
    X_train = X_train[:rows_count]
    Y_train = Y_train[:rows_count]

    dim = 28 * 28
    X_train = X_train.reshape(len(X_train), dim)

    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train,
                                                        test_size=0.3,
                                                        random_state=
                                                        random_state)

    # Random Forrest
    forrest = RandomForestClassifier(criterion='gini', min_samples_leaf=10,
                                     max_depth=20, n_estimators=10,
                                     random_state=random_state)
    clf_forrest = OneVsRestClassifier(forrest).fit(x_train, y_train)
    y_pred = clf_forrest.predict(x_test)

    # Get correctly predicted for Random Forrest from confusion matrix
    conf_matrx = confusion_matrix(y_test, y_pred)
    pred_correctly = 0
    for i in range(len(conf_matrx)):
        pred_correctly += conf_matrx[i][i]

    print('Random forrest for all classes = ' + str(pred_correctly))

    # Get image probability to predicted class via Random Forrest
    image = data.loc[image_id]
    image_rsh = image.to_numpy().reshape(1, -1)
    image_lbl = clf_forrest.predict(image_rsh)
    print('Image with id ' + str(image_id) + ' prediction probability = '
          + str(clf_forrest.predict_proba(image_rsh)[:, image_lbl][0][0]))

    # Get PCA and transform data
    pca = PCA(n_components=princip_comps, svd_solver='full')
    x_train_transf = pca.fit_transform(x_train)
    x_test_transf = pca.transform(x_test)

    # Decision Tree
    tree = DecisionTreeClassifier(criterion='gini', min_samples_leaf=10,
                                  max_depth=20, random_state=random_state)
    clf_tree = OneVsRestClassifier(tree).fit(x_train_transf, y_train)
    y_pred = clf_tree.predict(x_test_transf)

    # Get correctly predicted for Decision Tree from confusion matrix
    conf_matrx = confusion_matrix(y_test, y_pred)
    pred_correctly = 0
    for i in range(len(conf_matrx)):
        pred_correctly += conf_matrx[i][i]

    print('Decision tree for all classes = ' + str(pred_correctly))

    # Get image probability to predicted class via Decision Tree
    image_transf = pca.transform(image_rsh)
    image_lbl = clf_tree.predict(image_transf)
    print('Image with id ' + str(image_id) + ' prediction probability = '
          + str(clf_tree.predict_proba(image_transf)[:, image_lbl][0][0]))
