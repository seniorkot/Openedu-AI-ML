import pandas as pd


def print_results(cities_drop: list, index_col: str):
    data = pd.read_csv('exercise_1/data/salary_and_population.csv',
                       encoding='utf8', index_col=index_col).drop(cities_drop)

    print('Mean = ' + str(data['AVG_Salary'].mean()))

    print('Median = ' + str(data['AVG_Salary'].median()))

    print('Variance = ' + str(data['AVG_Salary'].var(ddof=0)))

    print('Standard deviation = ' + str(data['AVG_Salary'].std(ddof=0)))
