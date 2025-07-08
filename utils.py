# Импорты
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Внутренние модули проекта
from data_preprocessing import DATE_FORMAT, STRANGE_INTERVAL
import random
import time
import folium


def route_info(point1: dict, point2: dict):
    """
    Вывод информации об отрезке между двумя точками маршрута.

    :param point1: словарь с данными первой точки маршрута
    :param point2: словарь с данными второй точки маршрута
    """

    # Парсим дату из строки в объект datetime
    start_datetime = datetime.strptime(point1['datetime'], DATE_FORMAT)
    end_datetime = datetime.strptime(point2['datetime'], DATE_FORMAT)

    # Выводим модель автомобиля (если есть различие — покажем обе модели)
    models = {point1['vehicle']['model']['name'], point2['vehicle']['model']['name']}
    print(f"Model: {models}")

    # Вывод координат начала и конца отрезка
    print(f"Longitude: {point1['location']['longitude']} -> {point2['location']['longitude']}")
    print(f"Latitude:  {point1['location']['latitude']}  -> {point2['location']['latitude']}")

    # Временные метки
    print(f"Start: {start_datetime}; End: {end_datetime}")

    # Длительность отрезка
    print(f"Time difference: {end_datetime - start_datetime}")

    # Расстояние и скорость
    distance = route_length(point1, point2)
    print(f"Distance: {distance:.6f} (условных единиц)")
    print(f"Velocity: {point1['velocity']}, {point2['velocity']}\n")


def route_length(point1: dict, point2: dict) -> float:
    """
    Вычисляет "расстояние" между двумя точками маршрута.
    Здесь используется евклидово расстояние между широтой и долготой.

    :param point1: первая точка маршрута
    :param point2: вторая точка маршрута
    :return: евклидово расстояние между точками
    """

    # Широта и долгота
    lat1 = point1['location']['latitude']
    lon1 = point1['location']['longitude']
    lat2 = point2['location']['latitude']
    lon2 = point2['location']['longitude']

    # Евклидово расстояние между точками (упрощённый подход)
    return ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5


def routes_visualization(routes: list):
    """
    Визуализирует маршруты ТС на графике X-Y (широта vs долгота).
    Аномальные участки (длиннее STRANGE_INTERVAL) выводятся в консоль.

    :param routes: список маршрутов, каждый маршрут — список точек
    """

    for route in routes:
        x, y = [], []
        for i in range(1, len(route)):
            current_point = route[i]
            previous_point = route[i - 1]

            # Рассчитываем расстояние между соседними точками
            distance = route_length(previous_point, current_point)

            # Логируем аномалии
            if distance > STRANGE_INTERVAL:
                print(f"[INFO] Обнаружено аномальное перемещение между точками:")
                route_info(previous_point, current_point)

            # Сохраняем координаты для графика
            x.append(current_point['location']['longitude'])
            y.append(current_point['location']['latitude'])

        # Строим график маршрута
        plt.plot(x, y, marker='o', linestyle='-')

    plt.title("Маршруты ТС на плоскости (долгота, широта)")
    plt.xlabel("Долгота")
    plt.ylabel("Широта")
    plt.grid(True)
    plt.show()


def routes_in_time_visualization(routes):
    """
    Визуализация маршрутов в координатах: время (ось X), широта (ось Y).

    :param routes: список маршрутов, где каждый маршрут — список точек
    """

    for route in routes:
        times = [datetime.strptime(point['datetime'], DATE_FORMAT) for point in route]
        lats = [point['location']['latitude'] for point in route]

        plt.plot(times, lats, marker='o', linestyle='-')

    plt.title("Маршруты ТС во времени (время, широта)")
    plt.xlabel("Время")
    plt.ylabel("Широта")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def generate_random_hex_color() -> tuple[str, str]:
    """
    Генерирует два случайных цвета в HEX-формате.

    :return: кортеж из двух цветов — основной и для предсказаний
    """
    r = random.randint(0, 200)
    g = random.randint(0, 200)
    b = random.randint(0, 200)

    route_color = f"#{r:02x}{g:02x}{b:02x}"
    prediction_color = f"#{g:02x}{b:02x}{r:02x}"

    return route_color, prediction_color


def evaluation_visualization(x_test, y_test, scaler, model):
    """
    Визуализирует реальный маршрут и предсказание на карте с помощью Folium.

    :param x_test: тестовые данные (предыдущие точки)
    :param y_test: истинные значения (реальные следующие точки)
    :param scaler: нормализатор (MinMaxScaler)
    :param model: обученная модель машинного обучения
    """

    # Возвращаем данные к оригинальному масштабу
    y_test_descaled = scaler.inverse_transform(y_test)
    predicted_points = real_prediction(x_test[0], model, num=50)

    # Создаём карту
    m = folium.Map(location=y_test_descaled[0][2:4], zoom_start=13)
    route_color, prediction_color = generate_random_hex_color()

    # Добавляем точки на карту
    for i in range(len(predicted_points) - 1):
        current_point = y_test_descaled[i][2:4]
        next_point = y_test_descaled[i + 1][2:4]
        pred_point = predicted_points[i][2:4]
        next_pred_point = predicted_points[i + 1][2:4]

        # Метки на карте
        folium.Marker(
            location=current_point,
            icon=folium.Icon(color='blue'),
            popup=f'lat: {current_point[0]:.5f}\n'
                  f'lon: {current_point[1]:.5f}\n'
                  f'hours: {y_test_descaled[i][0]:.0f}\n'
                  f'minutes: {y_test_descaled[i][1]:.0f}'
        ).add_to(m)

        folium.Marker(
            location=pred_point,
            icon=folium.Icon(color='red'),
            popup=f'lat: {pred_point[0]:.5f}\n'
                  f'lon: {pred_point[1]:.5f}\n'
                  f'hours: {predicted_points[i][0]:.0f}\n'
                  f'minutes: {predicted_points[i][1]:.0f}'
        ).add_to(m)

        folium.Marker(location=next_point, icon=folium.Icon(color='green')).add_to(m)
        folium.Marker(location=next_pred_point, icon=folium.Icon(color='purple')).add_to(m)

        # Путь
        if y_test_descaled[i][6] != 0:
            folium.PolyLine([pred_point, next_pred_point], color=prediction_color).add_to(m)
            folium.PolyLine([current_point, next_point], color=route_color).add_to(m)

        if y_test_descaled[i][6] == 2:
            route_color, prediction_color = generate_random_hex_color()

    # Сохраняем карту в HTML
    m.save('prediction.html')


def real_prediction(start, model, num: int) -> list[list]:
    """
    Прогнозирует маршрут длиной `num` шагов, начиная с точки `start`.
    Каждое следующее значение предсказывается на основе предыдущего.

    :param start: стартовая точка (в нормализованном виде)
    :param model: обученная модель прогнозирования
    :param num: количество шагов для прогноза
    :return: список предсказанных точек (в не нормализованном виде)
    """

    real_start = start[:1].copy()
    print("[DEBUG] Начальная точка:", real_start)

    prediction = [real_start[0]]
    for i in range(num):
        # Предсказываем следующую точку
        current_point = model.predict(real_start)
        prediction.append(current_point[0])

        # Обновляем текущую точку
        real_start = current_point

    return prediction