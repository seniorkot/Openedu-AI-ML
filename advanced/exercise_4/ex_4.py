from imutils import paths
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import cv2
import os


def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None,
                        bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def print_results(c: float, random_state: int, criterion: str,
                  max_leaf_nodes: int, min_samples_leaf: int,
                  n_estimators: int, cv: int, images: list):
    image_paths = sorted(list(paths.list_images('exercise_2/data/train')))
    train_data = []
    labels = []

    for (i, imagePath) in enumerate(image_paths):
        image = cv2.imread(imagePath, 1)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        hist = extract_histogram(image)
        train_data.append(hist)
        labels.append(label)

    y = [1 if x == 'cat' else 0 for x in labels]

    # Decision Tree
    tree = DecisionTreeClassifier(criterion=criterion,
                                  min_samples_leaf=min_samples_leaf,
                                  max_leaf_nodes=max_leaf_nodes,
                                  random_state=random_state)
    bagging = BaggingClassifier(tree, n_estimators=n_estimators,
                                random_state=random_state)
    bagging.fit(train_data, y)

    # SVM
    svm = LinearSVC(random_state=random_state, C=c)
    svm.fit(train_data, y)

    # Random Forest
    forest = RandomForestClassifier(n_estimators=n_estimators,
                                    criterion=criterion,
                                    min_samples_leaf=min_samples_leaf,
                                    max_leaf_nodes=max_leaf_nodes,
                                    random_state=random_state)
    forest.fit(train_data, y)

    lr = LogisticRegression(solver='lbfgs', random_state=random_state)

    # Use Cross-Validation
    base_estimators = [('SVM', svm), ('Bagging DT', bagging),
                       ('DecisionForest', forest)]
    sclf = StackingClassifier(estimators=base_estimators, final_estimator=lr,
                              cv=cv)
    sclf.fit(train_data, y)

    print('Accuracy: ' + str(sclf.score(train_data, y)))

    for image in images:
        single_image = cv2.imread('exercise_2/data/test/' + image)
        histt = extract_histogram(single_image)
        histt2 = histt.reshape(1, -1)
        print(image + ' pred: ' + str(sclf.predict_proba(histt2)))
