from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
import pandas as pd
import graphviz


def print_results(lines_extract: int, criterion: str,
                  max_leaf_nodes: int, min_samples_leaf: int,
                  random_state: int, patients: list):
    df = pd.read_csv('exercise_3_2/data/diabetes.csv')

    task_data = df.head(lines_extract)

    print('Class [0, 1] len: ['
          + str(len(task_data[task_data['Outcome'] == 0])) + ', '
          + str(len(task_data[task_data['Outcome'] == 1])) + ']')

    train = task_data.head(int(len(task_data) * 0.8))
    test = task_data.tail(int(len(task_data) * 0.2))

    features = list(train.columns[:8])
    x = train[features]
    y = train['Outcome']

    tree = DecisionTreeClassifier(criterion=criterion,
                                  min_samples_leaf=min_samples_leaf,
                                  max_leaf_nodes=max_leaf_nodes,
                                  random_state=random_state)
    clf = tree.fit(x, y)
    print('Tree max depth: ' + str(clf.tree_.max_depth))

    features = list(test.columns[:8])
    x = test[features]
    y_true = test['Outcome']
    y_pred = clf.predict(x)

    print('Accuracy score: ' + str(accuracy_score(y_true, y_pred)))
    print('F1 score: ' + str(f1_score(y_true, y_pred, average='macro')))

    for patient in patients:
        print('Patient ' + str(patient) + ': '
              + str(clf.predict([df.loc[patient, features].tolist()])[0]))

    # Convert to png and view
    columns = list(x.columns)
    export_graphviz(clf, out_file='exercise_3_2/out/tree.dot',
                    feature_names=columns,
                    class_names=['0', '1'],
                    rounded=True, proportion=False,
                    precision=2, filled=True, label='all')
