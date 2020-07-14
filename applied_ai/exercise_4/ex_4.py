import pandas as pd


def print_results(tests: list):
    data = pd.read_csv('exercise_4/data/tests.csv', encoding='utf8',
                       index_col='name')

    for test in tests:
        test_value = data.loc[test]

        print('\n' + test + ':')
        print('is_anomally: ' + test_value['is_anomally'])
        print('NO: ' + str(test_value['coef_no']))
        print('YES: ' + str(test_value['coef_yes']))
