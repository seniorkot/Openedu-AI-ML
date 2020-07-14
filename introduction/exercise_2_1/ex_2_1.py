import pandas as pd
from sklearn.linear_model import LinearRegression


def print_results(data: pd.DataFrame):

    # Print X mean
    print('Xmean = ' + str(data['X'].mean()))

    # Print Y mean
    y_mean = data['Y'].mean()
    print('Ymean = ' + str(y_mean))

    X = pd.DataFrame(data['X'])
    y = pd.DataFrame(data['Y'])

    reg = LinearRegression().fit(X, y)

    # Print q0
    q0 = reg.intercept_[0]
    print('q0 = ' + str(q0))

    # Print q1
    q1 = reg.coef_[0][0]
    print('q1 = ' + str(q1))

    num = sum([(n[1] - q0 - q1*n[0])**2 for n in zip(data['X'], data['Y'])])

    den = sum([(y - y_mean)**2 for y in data['Y']])

    # Print R2
    print('R2 = ' + str(1 - (num / den)))

