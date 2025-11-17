"""
Пробная программа с поиском критических значений углов визирования
Модуль для анализа зависимости расстояния до спутника от угла визирования
"""

import math
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from pyorbital.orbital import Orbital
# Локальные модули
from calc_cord import get_xyzv_from_latlon
from read_TBF import read_tle_base_file

# Константы для расчета
EARTH_RADIUS = 6374.140  # радиус Земли в километрах
TAU = 10e-6              # Длительность импульса (с)
C_LIGHT = 3e5            # Скорость света (км/с)

def get_lat_lon_sgp(tle_1: str, tle_2: str, utc_time: datetime) -> tuple:
    """
    Получает географические координаты спутника по TLE
    
    Параметры:
        tle_1 (str): Первая строка двухстрочного TLE
        tle_2 (str): Вторая строка двухстрочного TLE
        utc_time (datetime): Время в формате UTC
    
    Возвращает:
        tuple: (Долгота, Широта, Высота) в градусах и километрах
    """
    orb = Orbital("N", line1=tle_1, line2=tle_2)
    return orb.get_lonlatalt(utc_time)

def calculate_theoretical_bounds(H_ORBIT: float):
    """Вычисление теоретических границ рабочей зоны"""
    theta_max = math.degrees(math.asin(EARTH_RADIUS/(EARTH_RADIUS + H_ORBIT)))
    R_min = (C_LIGHT * TAU) / 2  # в км
    return R_min, theta_max

def get_position(tle_1: str, tle_2: str, utc_time: datetime) -> tuple:
    """
    Вычисляет положение и скорость спутника в инерциальной системе координат
    
    Параметры:
        tle_1 (str): Первая строка TLE
        tle_2 (str): Вторая строка TLE
        utc_time (datetime): Временная метка
    
    Возвращает:
        tuple: (X, Y, Z, Vx, Vy, Vz) координаты (км) и скорость (км/с)
    """
    orb = Orbital("N", line1=tle_1, line2=tle_2)
    R_s, V_s = orb.get_position(utc_time, False)
    return (*R_s, *V_s)

def calculate_orbital_data(tle_1: str, tle_2: str, dt_start: datetime, 
                         dt_end: datetime, delta: timedelta, pos_gt: tuple) -> tuple:
    """
    Основная функция расчета орбитальных параметров
    
    Параметры:
        tle_1, tle_2 (str): Двухстрочный TLE
        dt_start (datetime): Начальное время расчета
        dt_end (datetime): Конечное время расчета
        delta (timedelta): Шаг расчета
        pos_gt (tuple): Координаты наземного объекта (широта, долгота, высота)
    
    Возвращает:
        tuple: Два списка - расстояния R0 (км) и углы визирования (градусы)
    """
    lat_t, lon_t, alt_t = pos_gt
    current_time = dt_start
    R_0_list = []
    y_grad_list = []
    H_list = []

    while current_time < dt_end:
        try:
            # Получение координат спутника
            X_s, Y_s, Z_s, _, _, _ = get_position(tle_1, tle_2, current_time)
            
            # Получение координат объекта
            pos_it, _ = get_xyzv_from_latlon(current_time, lon_t, lat_t, alt_t)
            X_t, Y_t, Z_t = pos_it

            # Вычисление расстояний
            delta_X = X_s - X_t
            delta_Y = Y_s - Y_t
            delta_Z = Z_s - Z_t
            R_0 = np.linalg.norm([delta_X, delta_Y, delta_Z])
            R_s = np.linalg.norm([X_s, Y_s, Z_s])
            R_e = np.linalg.norm([X_t, Y_t, Z_t])

            # Расчет угла визирования
            numerator = R_0**2 + R_s**2 - R_e**2
            denominator = 2 * R_0 * R_s
            
            if denominator == 0:
                continue

            cos_theta = numerator / denominator
            if not (-1 <= cos_theta <= 1):
                continue

            y_rad = math.acos(cos_theta)
            y_grad = math.degrees(y_rad)
            H_list.append(R_s - EARTH_RADIUS)

            # Фильтрация данных
            if 20 < y_grad < 70 and R_0 < R_e:  # Расширенный диапазон для анализа
                R_0_list.append(R_0)
                y_grad_list.append(y_grad)

        except Exception as e:
            print(f"Ошибка расчета для времени {current_time}: {str(e)}")
        
        current_time += delta

    return R_0_list, y_grad_list, H_list

def plot_orbital_data(R_0: list, y_grad: list, H_ORBIT: float, save_path: str = None):
    """
    Визуализация данных и сохранение графика
    
    Параметры:
        R_0 (list): Список расстояний
        y_grad (list): Список углов визирования
        H_ORBIT (float): Высота орбиты спутника
        save_path (str): Путь для сохранения графика
    """
    if not R_0 or not y_grad:
        print("Нет данных для построения графика")
        return

    # Конвертация в numpy массивы
    y_array = np.array(y_grad)
    R_array = np.array(R_0)
    
    # Расчет теоретических границ
    R_min_theory, theta_max_theory = calculate_theoretical_bounds(H_ORBIT)
    
    # Фильтрация данных
    mask = (y_array >= 24) & (y_array <= 55)
    R_filtered = R_array[mask]
    
    if len(R_filtered) == 0:
        print("Нет данных в рабочей зоне")
        return
    
    # Экспериментальные значения
    R_min_exp = np.min(R_filtered)
    R_max_exp = np.max(R_filtered)
    
    # Сортировка данных
    sort_idx = np.argsort(y_array)
    y_sorted, R_sorted = y_array[sort_idx], R_array[sort_idx]

    # Настройка графика
    plt.figure(figsize=(12, 7), dpi=100)
    plt.plot(y_sorted, R_sorted, 'b-', label='Экспериментальные данные')
    
    # Теоретические границы
    plt.axhline(R_min_theory, color='r', linestyle='--', label=f'Теоретический R_min: {R_min_theory:.1f} км')
    plt.axvline(theta_max_theory, color='g', linestyle='--', label=f'Теоретический θ_max: {theta_max_theory:.1f}°')
    
    # Экспериментальные границы
    plt.axhline(R_min_exp, color='orange', linestyle=':', label=f'Экспериментальный R_min: {R_min_exp:.1f} км')
    plt.axhline(R_max_exp, color='purple', linestyle=':', label=f'Экспериментальный R_max: {R_max_exp:.1f} км')

    # Настройки оформления
    plt.title('Сравнение теоретических и экспериментальных данных')
    plt.xlabel('Угол визирования, градусы')
    plt.ylabel('Расстояние R₀, км')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"График сохранен: {save_path}")
    
    plt.show()

def _test():
    """
    Тестовая функция для проверки работы модуля
    """
    try:
        # Загрузка TLE данных
        s_name, tle_1, tle_2 = read_tle_base_file(56756)
    except Exception as e:
        print(f"Ошибка загрузки TLE: {str(e)}")
        return

    # Параметры объекта наблюдения
    target_pos = (59.95, 30.316667, 12)
    
    # Настройки временного интервала
    start_time = datetime(2024, 2, 21, 3, 0, 0)
    time_delta = timedelta(seconds=30)
    end_time = start_time + timedelta(days=16)

    # Расчет орбитальных данных
    R_0, y_grad, H_list = calculate_orbital_data(
        tle_1, tle_2, start_time, end_time, time_delta, target_pos
    )
    
    if not H_list:
        print("Нет данных для анализа")
        return
    
    # Расчет средней высоты орбиты
    H_ORBIT = np.mean(H_list)
    print(f"Средняя высота орбиты: {H_ORBIT:.1f} км")

    # Визуализация результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/plot_{timestamp}.png"
    plot_orbital_data(R_0, y_grad, H_ORBIT, save_path)

if __name__ == "__main__":
    _test()