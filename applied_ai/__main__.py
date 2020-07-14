import datetime
from applied_ai.exercise_1 import *
from applied_ai.exercise_2 import *
from applied_ai.exercise_4 import *
from applied_ai.exercise_5 import *


def e_1(stop_name: str, vehicle_name: str, direction_type: str,
        route_num: int, curr_date: datetime.datetime):
    print('\nExercise 1:')
    ex_1.print_results(stop_name, vehicle_name, direction_type,
                       route_num, curr_date)


def e_2(audios: list):
    print('\nExercise 2:')
    ex_2.print_results(audios)


def e_4(tests: list):
    print('\nExercise 4:')
    ex_4.print_results(tests)


def e_5(data_url: str, tok_freq: int, tok_gr: int):
    print('\nExercise 5:')
    ex_5.print_results(data_url, tok_freq, tok_gr)


# Ex.5 English version
def e_5_en(data_url: str, tok_freq: int, tok_gr: int):
    print('\nExercise 5:')
    ex_5_en.print_results(data_url, tok_freq, tok_gr)


def example():
    print("===EXAMPLE===")

    # Exercise 1
    stop_name = 'УЛ. ЛЕНИНА УГ. ЧКАЛОВСКОГО ПР. [1]'
    vehicle_name = 'АВТОБУС'
    direction_type = 'ПРЯМОЕ'
    route_num = 1
    curr_date = datetime.datetime.fromisoformat('2019-09-09 18:33')
    e_1(stop_name, vehicle_name, direction_type, route_num, curr_date)

    # Exercise 2
    audios = [42, 64, 76, 23, 61, 56, 53]
    e_2(audios)

    # Exercise 4
    tests = ['test', 'test2', '101010']
    e_4(tests)

    # Exercise 5
    data_url = 'http://az.lib.ru/n/nekrasow_n_a/text_1840_pevitza.shtml'
    tok_freq = 50
    tok_gr = 100
    e_5(data_url, tok_freq, tok_gr)


if __name__ == "__main__":
    example()
