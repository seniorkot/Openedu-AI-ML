import pandas as pd
import math
from sklearn.linear_model import LogisticRegression


def print_results(data: pd.DataFrame, new_star: list, k: int):
    print('Count = ' + str(data['MIP'].count()))

    print('Mean = ' + str(data['MIP'].mean()))

    normalized = (data - data.min()) / (data.max() - data.min())
    normalized['TARGET'] = data['TARGET']

    print('Normalized mean = ' + str(normalized['MIP'].mean()))

    X = pd.DataFrame(normalized.drop(['TARGET'], axis=1))
    y = pd.DataFrame(normalized['TARGET'])

    reg = LogisticRegression(random_state=2019, solver='lbfgs') \
        .fit(X, y.values.ravel())

    result = reg.predict_proba([new_star])
    print('Pulsar probability = ' + str(result[0][1]))

    calc_func = lambda x: math.sqrt(
        sum([(new_col - col) ** 2 for new_col, col in zip(new_star, x)]))
    distances = normalized.apply(
        lambda x: pd.Series([calc_func(x), x['TARGET']],
                            index=['dist', 'TARGET']), axis=1)
    nearest = distances.sort_values(by=['dist'])

    print('Nearest = ' + str(distances.min()['dist']))

    print('Mode = ' + str(int(nearest.head(k).mode().iloc[0]['TARGET'])))
