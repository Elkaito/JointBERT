"""Dictionary that stores the count for each intent appearing in a dataset"""


atisCountDict = {
    'atis_flight': 3308,
    'atis_airfare': 385,
    'atis_ground_service': 230,
    'atis_airline': 139,
    'atis_abbreviation': 130,
    'atis_aircraft': 70,
    'atis_flight_time': 45,
    'atis_quantity': 41,
    'atis_city': 18,
    'atis_airport': 17,
    'atis_distance': 17,
    'atis_capacity': 15,
    'atis_ground_fare': 15,
    'atis_flight_no': 12,
    'atis_meal': 6,
    'atis_restriction': 5,
    'atis_cheapest': 1,
    'atis_flight#atis_airfare': 19,
    'atis_airline#atis_flight_no': 2,
    'atis_ground_service#atis_ground_fare': 1,
    'atis_aircraft#atis_flight#atis_flight_no': 1
}

snipsCountDict = {
    'PlayMusic': 1913,
    'GetWeather': 1896,
    'BookRestaurant': 1881,
    'RateBook': 1876,
    'SearchScreeningEvent': 1852,
    'SearchCreativeWork': 1847,
    'AddToPlaylist': 1818
}

fbAlarmCountDict = {
    'alarm/set_alarm': 4816,
    'alarm/cancel_alarm': 2069,
    'alarm/show_alarm': 1142,
    'alarm/modify_alarm': 439,
    'alarm/snooze_alarm': 432,
    'alarm/time_left_on_alarm': 384
}

fbReminderCountDict = {
    'reminder/set_reminder': 4743,
    'reminder/cancel_reminder': 1151,
    'reminder/show_reminder': 1006
}

fbWeatherCountDict = {
    'weather/find': 3953,
    'weather/checkSunset': 55,
    'weather/checkSunrise': 35
}