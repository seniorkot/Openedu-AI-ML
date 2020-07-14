import pandas as pd
import numpy as np
from introduction.exercise_1 import *
from introduction.exercise_2_1 import *
from introduction.exercise_2_2 import *
from introduction.exercise_3 import *
from introduction.exercise_5 import *
from introduction.final import *


def e_1(cities_drop: list):
    print('\nExercise 1:')
    ex_1.print_results(cities_drop, 'Region_RU')


def e_1_en(cities_drop: list):
    print('\nExercise 1:')
    ex_1.print_results(cities_drop, 'Region_EN')


def e_2_1(data: pd.DataFrame):
    print('\nExercise 2.1:')
    ex_2_1.print_results(data)


def e_2_2(candies_pred: list, custom_candy: list):
    print('\nExercise 2.2:')
    ex_2_2.print_results(candies_pred, custom_candy)


def e_3(candies_drop: list, candies_pred: list):
    print('\nExercise 3:')
    ex_3.print_results(candies_drop, candies_pred)


def e_5(data: pd.DataFrame, centroid: np.ndarray, cluster: int):
    print('\nExercise 5:')
    ex_5.print_results(data, centroid, cluster)


def e_final(data: pd.DataFrame, new_star: list, k: int):
    print('\nExercise Final:')
    final.print_results(data, new_star, k)


def example():
    print("===EXAMPLE===")

    # Exercise 1
    cities_drop = ['Вологодская область', 'Мурманская обл']
    e_1(cities_drop)

    # Exercise 2.1
    data = pd.read_csv('exercise_2_1/data/example.csv', encoding='utf8')
    e_2_1(data)

    # Exercise 2.2
    candies_pred = ['One dime', 'Nik L Nip']
    custom_candy = [[0, 1, 0, 1, 1, 0, 1, 1, 1, 0.848, 0.594]]
    e_2_2(candies_pred, custom_candy)

    # Exercise 3
    candies_drop = ['Charleston Chew', 'Nerds', 'Nik L Nip']
    candies_pred = ['Tootsie Roll Midgies', 'Trolli Sour Bites']
    e_3(candies_drop, candies_pred)

    # Exercise 5
    data = pd.read_csv('exercise_5/data/example.csv', encoding='utf8',
                       index_col='Object')
    centroid = np.array([[8.0, 12.0], [10.57, 7.43], [9.5, 9.5]])
    cluster = 0
    e_5(data, centroid, cluster)

    # Final
    data = pd.read_csv('final/data/example.csv', encoding='utf8')
    new_star = [0.772, 0.204, 0.137, 0.55, 0.316, 0.221, 0.19, 0.927]
    e_final(data, new_star, 5)


if __name__ == "__main__":
    example()
