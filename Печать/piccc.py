import math
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from pyorbital.orbital import Orbital
from scipy.optimize import fsolve

from calc_cord import get_xyzv_from_latlon
from read_TBF import read_tle_base_file

# Константы для расчета
EARTH_RADIUS = 6378.140  # Экваториальный радиус Земли в км
H_ORBIT = 500.0          # Высота орбиты в км
TAU = 10e-6              # Длительность импульса (с)
C_LIGHT = 3e5            # Скорость света (км/с)

def calculate_theoretical_bounds():
    """Вычисление теоретических границ рабочей зоны"""
    R_earth_km = EARTH_RADIUS
    H_km = H_ORBIT

    def equation(theta_deg, gamma):
        theta = np.radians(theta_deg)
        return np.sin(theta + gamma) - (R_earth_km / (R_earth_km + H_km)) * np.sin(theta)

    # Решение уравнений для граничных углов
    gamma_min = fsolve(lambda gamma: equation(24, gamma), 0.1)[0]
    gamma_max = fsolve(lambda gamma: equation(55, gamma), 0.1)[0]

    # Расчет геометрических границ
    R_min_geom = np.sqrt((R_earth_km + H_km)**2 + R_earth_km**2 - 
                     2*R_earth_km*(R_earth_km + H_km)*np.cos(gamma_min))
    
    R_max_geom = np.sqrt((R_earth_km + H_km)**2 + R_earth_km**2 - 
                     2*R_earth_km*(R_earth_km + H_km)*np.cos(gamma_max))

    # Учет задержки сигнала
    R_min_delay = (C_LIGHT * TAU) / 2  # в км
    R_min = max(R_min_geom, R_min_delay)

    # Генерация кривой зависимости
    gamma_values = np.linspace(gamma_min, gamma_max, 100)
    theta_values = []
    R_values = []

    for gamma in gamma_values:
        theta = fsolve(lambda theta: equation(theta, gamma), 24)[0]
        theta_values.append(theta)
        R = np.sqrt((R_earth_km + H_km)**2 + R_earth_km**2 - 
                 2*R_earth_km*(R_earth_km + H_km)*np.cos(gamma))
        R_values.append(R)

    return R_min, R_max_geom, R_values, theta_values

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
    # Распаковка координат наземного объекта
    lat_t, lon_t, alt_t = pos_gt
    
    # Инициализация списков для результатов
    R_0_list = []
    y_grad_list = []
    
    # Генерация временных меток с заданным шагом
    current_time = dt_start
    
    # Основной цикл расчетов
    while current_time < dt_end:
        dt = current_time
        # Получение координат спутника в инерциальной системе
        X_s, Y_s, Z_s, _, _, _ = get_position(tle_1, tle_2, dt)
        
        # Расчет координат наземного объекта в инерциальной системе
        pos_it, _ = get_xyzv_from_latlon(dt, lon_t, lat_t, alt_t)
        X_t, Y_t, Z_t = pos_it
        
        # Расчет расстояний:
        # R0 - расстояние между спутником и объектом
        delta_X = X_s - X_t
        delta_Y = Y_s - Y_t
        delta_Z = Z_s - Z_t
        R_0 = math.sqrt(delta_X**2 + delta_Y**2 + delta_Z**2)
        
        # Rs - расстояние от центра Земли до спутника
        R_s = math.sqrt(X_s**2 + Y_s**2 + Z_s**2)
        
        # Re - расстояние от центра Земли до объекта
        R_e = math.sqrt(X_t**2 + Y_t**2 + Z_t**2)
        
        # Расчет угла визирования по формуле косинусов
        try:
            # Числитель и знаменатель для формулы
            numerator = R_0**2 + R_s**2 - R_e**2
            denominator = 2 * R_0 * R_s
            
            # Защита от деления на ноль и некорректных значений
            if denominator == 0 or abs(numerator/denominator) > 1:
                continue
                
            y_rad = math.acos(numerator / denominator)
        except (ValueError, ZeroDivisionError):
            continue
        
        # Перевод радиан в градусы
        y_grad = math.degrees(y_rad)
        
        # Фильтрация данных по углу визирования и расстоянию
        if 15 < y_grad < 57 and R_0 < R_e:
            R_0_list.append(R_0)
            y_grad_list.append(y_grad)
        
        # Переход к следующему временному шагу
        current_time += delta
    
    return R_0_list, y_grad_list

def plot_orbital_data(R_0: list, y_grad: list, save_path: str = None):
    """Визуализация данных с теоретическими границами"""
    # Проверка наличия данных
    if not R_0 or not y_grad:
        print("Ошибка: Нет данных для построения графика")
        return

    # Конвертация в numpy массивы для векторных операций
    y_array = np.array(y_grad)
    R_array = np.array(R_0)
    
    # Параметры рабочей зоны (углы визирования)
    y_min_zone, y_max_zone = 24, 55
    
    # Фильтрация данных в рабочей зоне
    mask = (y_array >= y_min_zone) & (y_array <= y_max_zone)
    y_filtered = y_array[mask]
    R_filtered = R_array[mask]

    # Расчет теоретических значений
    R_min_theory, R_max_theory, R_theory, theta_theory = calculate_theoretical_bounds()
   
    if len(R_filtered) == 0:
        print("Нет данных в рабочей зоне")
        return
    
    # Расчет граничных значений расстояний
    R_min = np.min(R_filtered)
    R_max = np.max(R_filtered)
    
    # Определение углов при экстремальных расстояниях
    y_min_detected = y_filtered[np.argmin(R_filtered)]
    y_max_detected = y_filtered[np.argmax(R_filtered)]

    # Сортировка данных
    sort_idx = np.argsort(y_array)
    y_sorted, R_sorted = y_array[sort_idx], R_array[sort_idx]

    # Настройка графика
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    
    # График измерений для Кондор ФКА
    ax.plot(y_sorted, R_sorted, linewidth=2, color='#1f77b4', label='Реальные измерения')
    
  # Вертикальные линии границ рабочей зоны
    ax.axvline(
        x=y_min_zone, 
        color='r', 
        linestyle='--', 
        alpha=0.7, 
        label=f'ϒ_min = {y_min_zone}°'
    )
    ax.axvline(
        x=y_max_zone, 
        color='r', 
        linestyle='--', 
        alpha=0.7, 
        label=f'ϒ_max = {y_max_zone}°'
    )

    # Горизонтальные линии экстремальных расстояний
    ax.axhline(
        y=R_min, 
        color='g', 
        linestyle='-.', 
        alpha=0.7, 
        label=f'R_min = {R_min:.1f} км (ϒ={y_min_detected:.1f}°)'
    )
    ax.axhline(
        y=R_max, 
        color='b', 
        linestyle='-.', 
        alpha=0.7, 
        label=f'R_max = {R_max:.1f} км (ϒ={y_max_detected:.1f}°)'
    )

    # Заливка рабочей зоны
    ax.fill_betweenx(
        y=[R_min, R_max],
        x1=y_min_zone,
        x2=y_max_zone,
        color='lightgreen',
        alpha=0.1,
        label='Рабочая зона'
    )
    
    # Настройки оформления
    ax.set(xlabel='Угол визирования ϒ (°)', 
          ylabel='Расстояние R₀ (км)',
          title='Сравнение реальных и теоретических параметров',
          xlim=(20, 60),
          ylim=(min(R_min_theory, R_min)*0.95, max(R_max_theory, R_max)*1.05))
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    # Сохранение
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {save_path}")
    
    plt.show()

def _test():
    """
    Тестовая функция для проверки работы модуля
    """
    # Загрузка TLE из файла для спутника с номером 56756
    s_name, tle_1, tle_2 = read_tle_base_file(56756)
    
    # Координаты наземного объекта (Санкт-Петербург)
    target_pos = (59.95, 30.316667, 12)
    
    # Настройка временного интервала
    start_time = datetime(2024, 2, 21, 3, 0, 0)
    time_delta = timedelta(seconds=10)  # Шаг расчета 10 секунд
    end_time = start_time + timedelta(days=16)  # Период 16 дней

    # Основной расчет
    R_0, y_grad = calculate_orbital_data(
        tle_1, 
        tle_2, 
        start_time, 
        end_time, 
        time_delta, 
        target_pos
    )

    # Генерация имени файла с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/plot_{timestamp}.png"
    
    # Визуализация и сохранение
    plot_orbital_data(R_0, y_grad, save_path)

if __name__ == "__main__":
    # Точка входа при запуске скрипта
    _test()