from datetime import datetime
from datetime import timedelta
import pandas as pd


def print_results(stop_name: str, vehicle_name: str, direction_type: str,
                  route_num: int, curr_date: datetime):
    stops = pd.read_csv('exercise_1/data/stops.csv', encoding='utf8',
                        index_col='ID_STOP', na_values='-')
    vehicles = pd.read_csv('exercise_1/data/vehicle.csv', encoding='utf8',
                           index_col='ID_VEHICLE', na_values='-')
    directions = pd.read_csv('exercise_1/data/direction.csv', encoding='utf8',
                             index_col='ID_DIRECTION', na_values='-')
    routes = pd.read_csv('exercise_1/data/route_by_stops.csv',
                         encoding='utf8', na_values='-')
    tracks = pd.read_csv('exercise_1/data/track.csv', encoding='utf8',
                         na_values='-')
    tracks['STOP_TIME_REAL'] = pd.to_datetime(tracks['STOP_TIME_REAL'])

    # Step 1
    vehicle_id = vehicles[vehicles['VEHICLE_NAME'] == vehicle_name].index[0]
    direction_id = \
        directions[directions['DIRECTION_TYPE'] == direction_type].index[0]

    # Step 2
    route = routes[(routes['ROUTE_NUMBER'] == route_num)
                   & (routes['ID_STOP']
                      == stops[stops['STOP_NAME'] == stop_name].index[0])
                   & (routes['ID_VEHICLE'] == vehicle_id)
                   & (routes['ID_DIRECTION'] == direction_id)]
    distance_min = round(route['DISTANCE_BACK'].values[0] / (10000 / 60))

    # Step 3
    route = routes[(routes['ROUTE_NUMBER'] == route_num)
                   & (routes['ID_VEHICLE'] == vehicle_id)
                   & (routes['ID_DIRECTION'] == direction_id)
                   & (routes['STOP_NUMBER']
                      == route['STOP_NUMBER'].values[0] - 1)]
    stop_id = route['ID_STOP'].values[0]

    # Step 4
    track = tracks[(tracks['ROUTE_NUMBER'] == route_num)
                   & (tracks['ID_VEHICLE'] == vehicle_id)
                   & (tracks['ID_DIRECTION'] == direction_id)
                   & (tracks['ID_STOP'] == stop_id)
                   & (tracks['STOP_TIME_REAL'].notna())
                   & (tracks['STOP_TIME_REAL'] >=
                      (curr_date - timedelta(minutes=distance_min)))
                   & (tracks['STOP_TIME_REAL'] < curr_date)]

    track = track.sort_values('STOP_TIME_REAL')

    board_num = track['CARRIER_BOARD_NUM'].values[0]
    wait_time = (datetime.utcfromtimestamp(track['STOP_TIME_REAL']
                                           .values[0].astype(datetime) * 1e-9)
                 + timedelta(minutes=distance_min)) - curr_date

    print('Board number = ' + str(board_num))
    print('Waiting time = ', str(int(wait_time.seconds / 60)))
