from exam.introduction import *
from exam.advanced import *


def ex_introduction(tg_0: list, tg_1: list, random_state: int,
                    new_star: list, k: int):
    print('\nExam - Introduction:')
    introduction.print_results(tg_0, tg_1, random_state, new_star, k)


def ex_advanced(rows_count: int, random_state: int, image_id: int,
                princip_comps: int):
    print('\nExam - Advanced:')
    advanced.print_results(rows_count, random_state, image_id, princip_comps)


def example_advanced():
    print('===EXAMPLE ADVANCED===')

    rows_count = 3000
    random_state = 13
    image_id = 46
    princip_comps = 23
    ex_advanced(rows_count, random_state, image_id, princip_comps)


def example_basic():
    print('===EXAMPLE BASIC===')

    tg_0 = [97.4140625, 98.0078125]
    tg_1 = [47.4140625, 52.9296875]
    random_state = 9
    new_star = [0.269, 0.922, 0.142, 0.718, 0.183, 0.365, 0.42, 0.947]
    k = 20
    ex_introduction(tg_0, tg_1, random_state, new_star, k)


if __name__ == "__main__":
    example_advanced()
    example_basic()
