import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def print_results(candies_drop: list, candies_pred: list):
    data = pd.read_csv('exercise_3/data/candy-data.csv', encoding='utf8',
                       index_col='competitorname')

    train_data = data.drop(candies_drop)

    X = pd.DataFrame(train_data.drop(['winpercent', 'Y'], axis=1))
    y = pd.DataFrame(train_data['Y'])

    reg = LogisticRegression(random_state=2019, solver='lbfgs')\
        .fit(X, y.values.ravel())

    test_data = pd.read_csv("exercise_3/data/candy-test.csv", encoding='utf8',
                            index_col='competitorname')

    for candy in candies_pred:
        print(candy + ': ', end='')
        cnd = test_data.loc[candy].drop(['Y']).values
        probs = reg.predict_proba([cnd.tolist()])
        print('Probabilities = ' + str(probs))

    X_test = pd.DataFrame(test_data.drop(['Y'], axis=1))
    y_pred = reg.predict(X_test)
    y_true = test_data['Y'].to_frame().T.values.ravel()

    print('Recall = ' + str(metrics.recall_score(y_true, y_pred)))

    print('Precision = ' + str(metrics.precision_score(y_true, y_pred)))

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    print('AUC = ' + str(metrics.auc(fpr, tpr)))
