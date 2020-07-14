import pandas as pd
from sklearn.linear_model import LinearRegression


def print_results(candies_pred: list, custom_candy: list):
    data = pd.read_csv('exercise_2_2/data/candy-data.csv', encoding='utf8',
                       index_col='competitorname')

    train_data = data.drop(candies_pred)

    X = pd.DataFrame(train_data.drop(['winpercent', 'Y'], axis=1))
    y = pd.DataFrame(train_data['winpercent'])

    reg = LinearRegression().fit(X, y)

    # Predict for-each
    for candy in candies_pred:
        candy_series = data.loc[candy, :].to_frame().T
        prediction = \
            reg.predict(candy_series.drop(['winpercent', 'Y'], axis=1))
        print(candy + ' = ' + str(prediction))

    # Predict custom
    prediction = reg.predict(custom_candy)
    print('Custom = ' + str(prediction))
