import json
import hashlib
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import numpy as np
import requests

# Загрузка переменных окружения из файла .env
load_dotenv()

# Константы проекта
HASH_LENGTH = 4  # Длина хэша для обезличивания
STOP_TIME = timedelta(hours=4)  # Временной интервал между точками, после которого маршрут считается завершённым
DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%f%z'  # Формат даты из данных
STRANGE_INTERVAL = 0.2  # Не используется сейчас, возможно, будет позже


def date_processing(date_str: str) -> list:
    """
    Преобразует строку даты и времени в список чисел.

    Пример входной строки: '2024-12-01T04:00:38.821560+04:00'

    :param date_str: Строка с датой и временем
    :return: [год, месяц, день, час, минута, секунда]
    """
    dt = datetime.strptime(date_str, DATE_FORMAT)
    return [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second]


def hashing(filename: str) -> list[dict]:
    """
    Обезличивает данные в JSON-файле: заменяет реальный номер авто на хэш.

    :param filename: Путь к файлу с данными о машинах
    :return: Список обезличенных записей
    """
    with open(filename, encoding='utf-8') as f:
        data = json.load(f)

    secret_key = os.getenv("SECRET_KEY", "default_secret")  # Получаем секретный ключ
    hash_len = int(os.getenv("HASH_LENGTH", HASH_LENGTH))  # Длина хэша

    for car in data:
        # Убираем оригинальный номер и добавляем захешированное значение
        license_plate = car["vehicle"].pop('license_plate')
        hashed_plate = hashlib.shake_256((license_plate + secret_key).encode()).hexdigest(hash_len)
        key = hashlib.shake_256(b"license_plate").hexdigest(hash_len)
        car["vehicle"][key] = hashed_plate

    # Сохраняем обезличенные данные в отдельный файл
    with open(f"secure_data/secure_{os.path.basename(filename)}", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def api_data_hashing(url: str, offset: int, limit: int, license_plate: str) -> list[dict]:
    """
    Запрашивает данные из внешнего API и обезличивает их.

    :param url: Адрес API
    :param offset: Начальная точка выборки
    :param limit: Количество записей
    :param license_plate: Номер автомобиля для фильтрации
    :return: Список обезличенных маршрутов
    """
    params = {
        "offset": offset,
        "limit": limit,
        "license_plate": license_plate,
        "order_by": "event_datetime",
        "order": "DESC"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()

        secret_key = os.getenv("SECRET_KEY", "default_secret")
        hash_len = int(os.getenv("HASH_LENGTH", HASH_LENGTH))

        for car in data:
            real_plate = car["vehicle"].pop("license_plate")
            hashed_plate = hashlib.shake_256((real_plate + secret_key).encode()).hexdigest(hash_len)
            key = hashlib.shake_256(b"license_plate").hexdigest(hash_len)
            car["vehicle"][key] = hashed_plate

        # Сохраняем обезличенные данные
        filename = f"secure_data/secure_response_I{offset // 1000}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return data
    else:
        print(f"Ошибка при получении данных. Код ошибки: {response.status_code}")
        return []


def light_weight_routes(routes: list) -> list[list[dict]]:
    """
    Упрощает структуру данных: удаляет лишние поля, оставляя только важные.

    :param routes: Список маршрутов (каждый маршрут — список словарей)
    :return: Легковесная версия маршрутов
    """
    keys_to_keep = ['datetime', 'location', 'driving_direction', 'velocity']

    new_routes_list = []
    for route in routes:
        new_route = [
            {k: point[k] for k in keys_to_keep}
            for point in route
        ]
        new_routes_list.append(new_route)

    # Сохраняем в файл
    with open("routes/lightweight_routes.json", 'w', encoding='utf-8') as f:
        json.dump(new_routes_list, f, ensure_ascii=False, indent=2)

    return new_routes_list


def define_routes(data: list) -> list[list[dict]]:
    """
    Разделяет сырые данные на логические маршруты.
    Маршрут начинается с новой точки, если временной промежуток между точками больше STOP_TIME.

    :param data: Список точек маршрутов (неотсортированных)
    :return: Список маршрутов, каждый из которых — список точек
    """
    # Сортируем данные по времени
    sorted_data = sorted(data, key=lambda x: x['datetime'])

    routes = []  # Результат — список маршрутов
    start_index = 0
    start_time = datetime.strptime(sorted_data[0]['datetime'], DATE_FORMAT)

    for i in range(1, len(sorted_data)):
        current_time = datetime.strptime(sorted_data[i]['datetime'], DATE_FORMAT)
        time_diff = current_time - start_time

        if time_diff >= STOP_TIME:
            routes.append(sorted_data[start_index:i])
            start_index = i
        start_time = current_time

    # Добавляем последний маршрут
    routes.append(sorted_data[start_index:])
    return routes


def json_to_numpy_route(route: list[dict], features: int = 7) -> np.ndarray:
    """
    Преобразует один маршрут в NumPy массив для обучения модели.

    Формат выходного массива:
    [час, минута, широта, долгота, направление, скорость, тип точки]

    Тип точки:
    - 0 — начальная точка
    - 1 — промежуточная точка
    - 2 — конечная точка

    :param route: Маршрут как список словарей
    :param features: Число признаков
    :return: NumPy массив размером (len(route), features)
    """
    np_route = np.zeros((len(route), features))
    for i, point in enumerate(route):
        # Извлекаем время и берем только час и минуту
        np_route[i][0:2] = date_processing(point['datetime'])[3:5]  # [час, минута]
        # Координаты
        np_route[i][2] = point['location']['latitude']
        np_route[i][3] = point['location']['longitude']
        # Дополнительные параметры
        np_route[i][4] = point['driving_direction']
        np_route[i][5] = point['velocity']
        # Тип точки
        if i == 0:
            category = 0
        elif i == len(route) - 1:
            category = 2
        else:
            category = 1
        np_route[i][6] = category

    return np_route


def create_sequences(data: np.ndarray, seq_length: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Создаёт обучающие пары (X, y) для моделей типа LSTM.

    X: Последовательность из seq_length точек
    y: Следующая точка

    :param data: Полный набор точек (NumPy array)
    :param seq_length: Длина последовательности
    :return: X, y
    """
    x, y = [], []

    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i])  # предыдущие точки
        y.append(data[i])  # следующая точка

    return np.array(x), np.array(y)


def create_dataset(routes: list[list[dict]], features: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """
    Преобразует маршруты в пары (X, y), где X — текущая точка, y — следующая точка.

    :param routes: Список маршрутов
    :param features: Число используемых признаков
    :return: x_train, y_train
    """
    x, y = [], []

    for route in routes:
        for i in range(len(route) - 1):
            x.append(route[i][:features])
            y.append(route[i + 1][:features])

    return np.array(x), np.array(y)


def get_lw_routes_from_files(files_dir: str) -> list[dict]:
    """
    Читает JSON-файлы из директории, обезличивает и сохраняет в формате маршрутов.

    :param files_dir: Путь к папке с исходными файлами
    :return: Список обработанных маршрутов
    """
    file_list = os.listdir(files_dir)
    lw_routes = []

    for filename in file_list:
        full_path = os.path.join(files_dir, filename)
        secure_data = hashing(full_path)

        lightweight_route = light_weight_routes(define_routes(secure_data))
        lw_routes.extend(lightweight_route)

    return lw_routes


def get_lw_routes_from_api(total_requests: int = 3) -> list[dict]:
    """
    Получает данные о маршрутах через API, обезличивает и сохраняет.

    :param total_requests: Число запросов к API
    :return: Список маршрутов
    """
    lw_routes = []

    for i in range(total_requests):
        secure_data = api_data_hashing(
            url='http://10.63.8.18:8080/v1/passages/filters/by-everything',
            offset=i * 1000,
            limit=1000,
            license_plate='X829KM763'
        )
        lightweight_route = light_weight_routes(define_routes(secure_data))
        lw_routes.extend(lightweight_route)

        # Сохраняем в файл
        with open(f"routes/response_{i}.json", 'w', encoding='utf-8') as f:
            json.dump(lightweight_route, f, ensure_ascii=False, indent=2)

    return lw_routes


def routes_files_concat(files_count: int = 3):
    """
    Объединяет несколько JSON-файлов с маршрутами в один общий файл.

    :param files_count: Число файлов
    """
    all_routes = []

    for i in range(files_count):
        with open(f"routes/response_{i}.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_routes.extend(data)

    # Сохраняем все маршруты в одном файле
    with open("routes/routesI.json", 'w', encoding='utf-8') as f:
        json.dump(all_routes, f, ensure_ascii=False, indent=2)

    return all_routes


def main():
    """
    Основная функция запуска программы.
    Получает данные, обрабатывает и сохраняет в виде облегчённых маршрутов.
    """
    print("[INFO] Получение и обработка маршрутов из API...")
    routes = get_lw_routes_from_api(3)

    print("[INFO] Сохранение объединённого датасета...")
    routes_files_concat(3)

    print("[INFO] Готово!")


if __name__ == '__main__':
    main()