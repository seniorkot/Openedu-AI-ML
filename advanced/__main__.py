from advanced.exercise_1_1 import *
from advanced.exercise_1_2 import *
from advanced.exercise_2 import *
from advanced.exercise_3_2 import *
from advanced.exercise_4 import *
from advanced.final import *


def e_1_1(filename: str, variance_lvl: float):
    print('\nExercise 1.1:')
    ex_1_1.print_results(filename, variance_lvl)


def e_1_2(scores: str, loadings: str):
    print('\nExercise 1.2:')
    ex_1_2.print_results(scores, loadings)


def e_2(c: float, random_state: int, qs: list, images: list):
    print('\nExercise 2:')
    ex_2.print_results(c, random_state, qs, images)


def e_3_2(lines_extract: int, criterion: str,
          max_leaf_nodes: int, min_samples_leaf: int,
          random_state: int, patients: list):
    print('\nExercise 3.2:')
    ex_3_2.print_results(lines_extract, criterion, max_leaf_nodes,
                         min_samples_leaf, random_state, patients)


def e_4(c: float, random_state: int, criterion: str, max_leaf_nodes: int,
        min_samples_leaf: int, n_estimators: int, cv: int, images: list):
    print('\nExercise 4:')
    ex_4.print_results(c, random_state, criterion, max_leaf_nodes,
                       min_samples_leaf, n_estimators, cv, images)


def e_final(variance_lvl: float, random_state: int,
            rf_class: int, lgr_class: int, dt_class: int, files: list):
    print('\nFinal:')
    final.print_results(variance_lvl, random_state,
                        rf_class, lgr_class, dt_class, files)


def example():
    print('===EXAMPLE===')

    # Exercise 1.1
    filename = 'exercise_1_1/data/14_25.csv'
    variance_lvl = 0.85
    e_1_1(filename, variance_lvl)

    # Exercise 1.2
    scores = 'exercise_1_2/data/X_reduced_536.csv'
    loadings = 'exercise_1_2/data/X_loadings_536.csv'
    e_1_2(scores, loadings)

    # Exercise 2
    c = 1.47
    random_state = 4
    qs = [152, 66, 123]
    images = ['cat.1020.jpg', 'cat.1010.jpg', 'dog.1046.jpg', 'dog.1013.jpg']
    e_2(c, random_state, qs, images)

    # Exercise 3.2
    lines_extract = 580
    criterion = 'entropy'
    max_leaf_nodes = 25
    min_samples_leaf = 15
    random_state = 2020
    patients = [749, 715, 718, 735]
    e_3_2(lines_extract, criterion, max_leaf_nodes,
          min_samples_leaf, random_state, patients)

    # Exercise 4
    c = 1.19
    random_state = 288
    criterion = 'entropy'
    max_leaf_nodes = 20
    min_samples_leaf = 10
    n_estimators = 21
    cv = 2
    images = ['dog.1025.jpg', 'cat.1027.jpg', 'cat.1006.jpg', 'cat.1002.jpg']
    e_4(c, random_state, criterion, max_leaf_nodes,
        min_samples_leaf, n_estimators, cv, images)

    # Final
    variance_lvl = 0.82
    random_state = 68
    rf_class = 3
    lgr_class = 8
    dt_class = 5
    files = ['file18', 'file15', 'file19']
    e_final(variance_lvl, random_state, rf_class, lgr_class, dt_class, files)


if __name__ == "__main__":
    example()
