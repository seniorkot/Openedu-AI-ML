import pandas as pd
import math
from sklearn.linear_model import LogisticRegression


def print_results(tg_0: list, tg_1: list, random_state: int,
                  new_star: list, k: int):
    data = pd.read_csv('introduction/data/pulsar_stars_new.csv',
                       encoding='utf-8')

    data_0 = data[data['MIP'].between(tg_0[0], tg_0[1], True)]
    data_0 = data_0[data_0['TG'] == 0]
    data_1 = data[data['MIP'].between(tg_1[0], tg_1[1], True)]
    data_1 = data_1[data_1['TG'] == 1]
    data = data_0.append(data_1)

    print('Number of rows = ' + str(len(data)))

    # Normalize
    normalized = (data - data.min()) / (data.max() - data.min())
    normalized['TG'] = data['TG']

    print('Normalized MIP mean = ' + str(normalized['MIP'].mean()))

    # Train Logistic Regression and Predict new star
    X = pd.DataFrame(normalized.drop(['TG'], axis=1))
    y = pd.DataFrame(normalized['TG'])

    reg = LogisticRegression(random_state=random_state, solver='lbfgs')\
        .fit(X, y.values.ravel())

    result = reg.predict_proba([new_star])
    print('Pulsar probability = ' + str(result[0][1]))

    # Calculate the nearest star
    calc_func = lambda x: math.sqrt(
        sum([(new_col - col) ** 2 for new_col, col in zip(new_star, x)]))
    distances = normalized.apply(
        lambda x: pd.Series([calc_func(x), x['TG']],
                            index=['dist', 'TG']), axis=1)
    nearest = distances.sort_values(by=['dist'])

    print('Nearest = ' + str(distances.min()['dist']))

    print('Pulsars nearby = '
          + str(len(nearest.head(k)[nearest.head(k)['TG'] == 1])))
