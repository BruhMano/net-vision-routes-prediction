import os
import tensorflow as tf
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
from data_preprocessing import json_to_numpy_route, create_dataset, create_sequences
from utils import evaluation_visualization

# Глобальные параметры модели
FEATURES = 7  # Количество признаков в данных
SEQ_LENGTH = 3  # Длина последовательности для LSTM
BATCH_SIZE = 128  # Размер батча для обучения
EPOCHS = 200  # Максимальное количество эпох обучения
VALIDATION_SPLIT = 0.8  # Доля данных для обучения (остальное для валидации)
LEARNING_RATE = 0.001  # Скорость обучения
FEATURES_WEIGHTS = [2.0, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0]  # Веса признаков для взвешенной MSE


def create_weighted_mse(feature_weights):
    """
    Создает кастомную функцию потерь с весами для разных признаков.

    Параметры:
        feature_weights (list): Список весов для каждого признака

    Возвращает:
        weighted_mse (function): Функция потерь с учетом весов признаков
    """
    feature_weights = tf.constant(feature_weights, dtype=tf.float32)
    feature_weights /= tf.reduce_sum(feature_weights)  # Нормировка весов

    def weighted_mse(y_true, y_pred):
        """
        Взвешенная MSE функция потерь.

        Параметры:
            y_true (tensor): Истинные значения
            y_pred (tensor): Предсказанные значения

        Возвращает:
            Средневзвешенную MSE по всем признакам
        """
        squared_errors = tf.square(y_true - y_pred)  # Квадраты ошибок
        weighted_squared_errors = squared_errors * feature_weights  # Умножение на веса
        return tf.reduce_mean(tf.reduce_sum(weighted_squared_errors, axis=1))  # Сумма по признакам и среднее по батчу

    return weighted_mse


def model_train(x_train, y_train, x_test, y_test, model_type, scaler):
    """
    Обучает модель в зависимости от выбранного типа.

    Параметры:
        x_train (ndarray): Обучающие данные
        y_train (ndarray): Целевые значения для обучения
        x_test (ndarray): Тестовые данные
        y_test (ndarray): Целевые значения для теста
        model_type (str): Тип модели ('dense', 'xgboost' или 'LSTM')
        scaler (MinMaxScaler): Объект для обратного масштабирования данных

    Возвращает:
        model: Обученную модель
    """
    if model_type == "dense":
        # Создание полносвязной нейронной сети
        model = Sequential([
            Dense(100, activation='relu', input_shape=(FEATURES,)),
            Dropout(0.3),  # Dropout для регуляризации
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dense(FEATURES)  # Выходной слой с количеством нейронов = количеству признаков
        ])

        weighted_mse = create_weighted_mse(FEATURES_WEIGHTS)
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss=weighted_mse)

        # Callbacks для ранней остановки и уменьшения learning rate
        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=0.001, patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7)
        ]

        # Обучение модели
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1
        )

        print(model.evaluate(x_test, y_test, batch_size=BATCH_SIZE))

    elif model_type == "xgboost":
        # Обучение модели XGBoost
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7
        )
        model.fit(x_train, y_train)

        # Оценка качества модели
        y_pred = model.predict(x_test)
        print(f'Средне-квадратическая ошибка на тестовых данных: {mean_squared_error(y_test, y_pred):.5f}')
        print(f'R^2 показатель на тестовых данных: {r2_score(y_test, y_pred):.5f}')
        return model

    elif model_type == "LSTM":
        # Создание LSTM модели
        model = Sequential([
            LSTM(64, return_sequences=False, input_shape=(SEQ_LENGTH, FEATURES)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(FEATURES)
        ])

        model.compile(optimizer='adam', loss='mse')
        history = model.fit(
            x_train,
            y_train,
            epochs=250,
            batch_size=64,
            verbose=1
        )

    # Предсказание и оценка результатов
    preds = model.predict(x_test)
    preds_original = scaler.inverse_transform(preds)  # Обратное масштабирование
    true_original = scaler.inverse_transform(y_test)

    # Расчет ошибок для координат
    lat_errors = np.abs(preds_original[:, 2] - true_original[:, 2])  # Ошибки по широте
    lon_errors = np.abs(preds_original[:, 3] - true_original[:, 3])  # Ошибки по долготе

    print(f'Средне-квадратическая ошибка: {mean_squared_error(true_original, preds_original):.5f}')
    print(f'R^2 показатель: {r2_score(true_original, preds_original):.5f}')
    print(f"Средняя ошибка по широте: {np.mean(lat_errors):.5f}")
    print(f"Средняя ошибка по долготе: {np.mean(lon_errors):.5f}")

    return model


def main(model_type: str, is_train: bool, visualization_len: int):
    """
    Основная функция для обучения и оценки модели.

    Параметры:
        model_type (str): Тип модели ('dense', 'xgboost' или 'LSTM')
        is_train (bool): Флаг обучения модели (True) или загрузки (False)
        visualization_len (int): Количество примеров для визуализации
    """
    # Загрузка данных из JSON файла
    with open(f"routes/routes.json") as f:
        routes_json = json.load(f)

    # Преобразование JSON в numpy массив
    routes = json_to_numpy_route(routes_json[0], FEATURES)
    scaler = MinMaxScaler()
    for i in range(1, len(routes_json)):
        routes = np.vstack([routes, json_to_numpy_route(routes_json[i], FEATURES)])

    # Масштабирование данных
    routes_scaled = scaler.fit_transform(routes)

    # Создание последовательностей для LSTM или обычного датасета
    if model_type == 'LSTM':
        x, y = create_sequences(routes_scaled, SEQ_LENGTH)
    else:
        x, y = create_dataset(routes_scaled, FEATURES)

    # Разделение на обучающую и тестовую выборки
    split_idx = int(VALIDATION_SPLIT * len(x))
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]

    # Определение расширения файла модели
    file_type = 'json' if model_type == 'xgboost' else 'keras'

    # Путь к файлу модели
    model_filename = f'models/routes_prediction_model_{model_type}.{file_type}'

    # Загрузка или обучение модели
    if os.path.isfile(model_filename) and not is_train:
        if model_type == "xgboost":
            model = XGBRegressor()
            model.load_model(model_filename)
        else:
            model = load_model(model_filename)
    else:
        model = model_train(x_train, y_train, x_test, y_test, model_type, scaler)
        # Сохранение модели
        if model_type == 'xgboost':
            model.save_model(model_filename)
        else:
            model.save(model_filename)

    # Визуализация результатов
    evaluation_visualization(x_test[:visualization_len], y_test[:visualization_len], scaler, model)


if __name__ == "__main__":
    # Запуск основной функции с параметрами
    main("xgboost", True, 30)