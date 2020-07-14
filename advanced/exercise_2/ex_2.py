from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from imutils import paths
from sklearn.metrics import f1_score
import numpy as np
import cv2
import os


def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins,
                        [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def print_results(c: float, random_state: int, qs: list, images: list):
    image_paths = sorted(list(paths.list_images('exercise_2/data/train')))
    data = []
    labels = []

    for (i, imagePath) in enumerate(image_paths):
        image = cv2.imread(imagePath, 1)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        hist = extract_histogram(image)
        data.append(hist)
        labels.append(label)

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    train_data, test_data, train_labels, test_labels = train_test_split(
        np.array(data), labels, test_size=0.25, random_state=random_state)

    model = LinearSVC(random_state=random_state, C=c)
    model.fit(train_data, train_labels)

    # Print coefficients
    for q in qs:
        print('q' + str(q) + ': ' + str(model.coef_[0][q]))

    predictions = model.predict(test_data)

    # Print F1 score
    f1 = f1_score(test_labels, predictions, average='macro')
    print('F1 score: ' + str(f1))

    for image in images:
        single_image = cv2.imread('exercise_2/data/test/' + image)
        histt = extract_histogram(single_image)
        histt2 = histt.reshape(1, -1)
        print(image + ' pred: ' + str(model.predict(histt2)))
