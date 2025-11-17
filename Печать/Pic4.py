"""
Модуль для анализа зависимости расстояния до спутника от угла визирования

Модель взята из статей, углы визирования взяты для КА Кондор ФКА
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from pyorbital.orbital import Orbital
from scipy.interpolate import UnivariateSpline

# Локальные модули
from calc_cord import get_xyzv_from_latlon
from read_TBF import read_tle_base_file

# Константы для расчета
EARTH_RADIUS = 6378.140  # Экваториальный радиус Земли в километрах
TAU = 10e-6              # Длительность импульса (с)
H_ORBIT = 506.9          # Высота орбиты (км)
C_LIGHT = 3e5            # Скорость света (км/с)

# Параметры рабочей зоны
GAMMA_MIN_DEG = 20       # Минимальный угол визирования для фильтрации (градусы)
GAMMA_WORK_ZONE_MIN = 24 # Минимальный угол рабочей зоны (градусы)
GAMMA_WORK_ZONE_MAX = 55 # Максимальный угол рабочей зоны (градусы)
GAMMA_PLOT_MAX = 60      # Максимальный угол для отображения на графике (градусы)

# Параметры по умолчанию
DEFAULT_SATELLITE_ID = 56756  # NORAD ID спутника Кондор ФКА
DEFAULT_TARGET_POS = (59.95, 30.316667, 12)  # Санкт-Петербург (широта, долгота, высота)

def get_lat_lon_sgp(tle_1: str, tle_2: str, utc_time: datetime) -> Tuple[float, float, float]:
    """
    Получает географические координаты спутника по TLE
    
    Параметры:
        tle_1: Первая строка двухстрочного TLE
        tle_2: Вторая строка двухстрочного TLE
        utc_time: Время в формате UTC
    
    Возвращает:
        Кортеж (долгота, широта, высота) в градусах и километрах
    """
    orb = Orbital("N", line1=tle_1, line2=tle_2)
    return orb.get_lonlatalt(utc_time)

def calculate_theoretical_bounds() -> Tuple[float, float]:
    """
    Вычисление теоретических границ рабочей зоны
    
    Возвращает:
        Кортеж (минимальное расстояние R_min, максимальный угол thetra_max)
    """
    thetra_max = np.degrees(np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + H_ORBIT)))
    R_min = (C_LIGHT * TAU) / 2  # в км (C_LIGHT уже в км/с, TAU в с)
    return R_min, thetra_max

def get_position(tle_1: str, tle_2: str, utc_time: datetime) -> Tuple[float, float, float, float, float, float]:
    """
    Вычисляет положение и скорость спутника в инерциальной системе координат
    
    Параметры:
        tle_1: Первая строка TLE
        tle_2: Вторая строка TLE
        utc_time: Временная метка
    
    Возвращает:
        Кортеж (X, Y, Z, Vx, Vy, Vz) координаты (км) и скорость (км/с)
    """
    orb = Orbital("N", line1=tle_1, line2=tle_2)
    R_s, V_s = orb.get_position(utc_time, False)
    return (*R_s, *V_s)

def calculate_orbital_data(
    tle_1: str, 
    tle_2: str, 
    dt_start: datetime, 
    dt_end: datetime, 
    delta: timedelta, 
    pos_gt: Tuple[float, float, float],
    theta_max: float
) -> Tuple[List[float], List[float]]:
    """
    Основная функция расчета орбитальных параметров
    
    Параметры:
        tle_1, tle_2: Двухстрочный TLE
        dt_start: Начальное время расчета
        dt_end: Конечное время расчета
        delta: Шаг расчета
        pos_gt: Координаты наземного объекта (широта, долгота, высота)
        theta_max: Максимальный угол визирования для фильтрации (градусы)
    
    Возвращает:
        Кортеж из двух списков: расстояния R0 (км) и углы визирования (градусы)
    """
    # Распаковка координат наземного объекта
    lat_t, lon_t, alt_t = pos_gt
    
    # Инициализация списков для результатов
    R_0_list: List[float] = []
    gamma_grad_list: List[float] = []
    
    # Генерация временных меток с заданным шагом
    current_time = dt_start
    
    # Основной цикл расчетов
    while current_time < dt_end:
        # Получение координат спутника в инерциальной системе
        X_s, Y_s, Z_s, _, _, _ = get_position(tle_1, tle_2, current_time)
        
        # Расчет координат наземного объекта в инерциальной системе
        pos_it, _ = get_xyzv_from_latlon(current_time, lon_t, lat_t, alt_t)
        X_t, Y_t, Z_t = pos_it
        
        # Расчет расстояний с использованием numpy для оптимизации
        # R0 - расстояние между спутником и объектом
        delta_pos = np.array([X_s - X_t, Y_s - Y_t, Z_s - Z_t])
        R_0 = np.linalg.norm(delta_pos)

        # Rs - расстояние от центра Земли до спутника
        R_s = np.linalg.norm([X_s, Y_s, Z_s])
        
        # Re - расстояние от центра Земли до объекта
        R_e = np.linalg.norm([X_t, Y_t, Z_t])
        
        # Расчет угла визирования по формуле косинусов
        denominator = 2 * R_0 * R_s
        
        # Защита от деления на ноль
        if denominator == 0 or R_0 == 0 or R_s == 0:
            current_time += delta
            continue
        
        numerator = R_0**2 + R_s**2 - R_e**2
        cos_gamma = numerator / denominator
        
        # Проверка на корректность значения арккосинуса
        if abs(cos_gamma) > 1.0:
            current_time += delta
            continue
        
        try:
            gamma_rad = np.arccos(cos_gamma)
        except (ValueError, ArithmeticError):
            current_time += delta
            continue
        
        # Перевод радиан в градусы
        gamma_grad = np.degrees(gamma_rad)
        
        # Фильтрация данных по углу визирования и расстоянию
        # Не проводим расчеты при угле визирования больше 57 градусов
        if (GAMMA_MIN_DEG < gamma_grad <= 57.0 and gamma_grad < theta_max and R_0 < R_e):
            R_0_list.append(float(R_0))
            gamma_grad_list.append(float(gamma_grad))
        
        # Переход к следующему временному шагу
        current_time += delta
    
    return R_0_list, gamma_grad_list

def plot_orbital_data(
    R_0: List[float], 
    gamma_grad: List[float], 
    save_path: Optional[str] = None
) -> None:
    """
    Визуализация данных и сохранение графика
    
    Параметры:
        R_0: Список расстояний (км)
        gamma_grad: Список углов визирования (градусы)
        save_path: Путь для сохранения графика (опционально)
    """
    # Проверка наличия данных
    if not R_0 or not gamma_grad:
        raise ValueError("Нет данных для построения графика")
    
    if len(R_0) != len(gamma_grad):
        raise ValueError("Длины массивов R_0 и gamma_grad не совпадают")

    # Конвертация в numpy массивы для векторных операций
    gamma_array = np.array(gamma_grad)
    R_array = np.array(R_0)
    
    # Фильтрация данных в рабочей зоне
    mask = (gamma_array >= GAMMA_WORK_ZONE_MIN) & (gamma_array <= GAMMA_WORK_ZONE_MAX)
    gamma_filtered = gamma_array[mask]
    R_filtered = R_array[mask]
    
    # Проверка наличия данных в рабочей зоне
    if len(R_filtered) == 0:
        raise ValueError(
            f"Нет данных в рабочей зоне ϒ={GAMMA_WORK_ZONE_MIN}°-{GAMMA_WORK_ZONE_MAX}°"
        )
    
    # Расчет граничных значений расстояний
    R_min = float(np.min(R_filtered))
    R_max = float(np.max(R_filtered))
    
    # Определение углов при экстремальных расстояниях
    gamma_min_detected = float(gamma_filtered[np.argmin(R_filtered)])
    gamma_max_detected = float(gamma_filtered[np.argmax(R_filtered)])

    # Сортировка данных для построения графика
    sorted_indices = np.argsort(gamma_array)
    gamma_sorted = gamma_array[sorted_indices]
    R_sorted = R_array[sorted_indices]

    # Сглаживание данных с помощью интерполяции
    # Используем UnivariateSpline для создания плавной кривой
    # Параметр s определяет степень сглаживания (меньше = более сглаженная линия)
    # Используем автоматический подбор параметра s на основе количества точек
    smoothing_factor = len(gamma_sorted) * np.var(R_sorted) * 0.1  # Настройка степени сглаживания
    
    try:
        spline = UnivariateSpline(gamma_sorted, R_sorted, s=smoothing_factor)
        # Создаем более плотную сетку точек для более плавной линии
        gamma_smooth = np.linspace(gamma_sorted.min(), gamma_sorted.max(), 
                                   max(500, len(gamma_sorted) * 5))
        R_smooth = spline(gamma_smooth)
    except Exception:
        # Если интерполяция не удалась, используем исходные данные
        gamma_smooth = gamma_sorted
        R_smooth = R_sorted

    # Настройка стиля графика
    style_name = 'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else (
        'seaborn' if 'seaborn' in plt.style.available else 'default'
    )
    plt.style.use(style_name)
    
    fig, ax = plt.subplots(
        figsize=(12, 7), 
        dpi=100, 
        facecolor='#f0f0f0'  # Цвет фона
    )

    # Построение основной линии графика (сглаженной)
    ax.plot(
        gamma_smooth, 
        R_smooth, 
        linewidth=2.5,
        color='#1f77b4',  # Стандартный синий цвет matplotlib
        label='Траектория измерений'
    )

    # Вертикальные линии границ рабочей зоны
    ax.axvline(
        x=GAMMA_WORK_ZONE_MIN, 
        color='r', 
        linestyle='--', 
        alpha=0.7, 
        label=f'ϒ_min = {GAMMA_WORK_ZONE_MIN}°'
    )
    ax.axvline(
        x=GAMMA_WORK_ZONE_MAX, 
        color='r', 
        linestyle='--', 
        alpha=0.7, 
        label=f'ϒ_max = {GAMMA_WORK_ZONE_MAX}°'
    )

    # Горизонтальные линии экстремальных расстояний
    ax.axhline(
        y=R_min, 
        color='g', 
        linestyle='-.', 
        alpha=0.7, 
        label=f'R_min = {R_min:.1f} км (ϒ={gamma_min_detected:.1f}°)'
    )
    ax.axhline(
        y=R_max, 
        color='b', 
        linestyle='-.', 
        alpha=0.7, 
        label=f'R_max = {R_max:.1f} км (ϒ={gamma_max_detected:.1f}°)'
    )

    # Заливка рабочей зоны
    ax.fill_betweenx(
        y=[R_min, R_max],
        x1=GAMMA_WORK_ZONE_MIN,
        x2=GAMMA_WORK_ZONE_MAX,
        color='lightgreen',
        alpha=0.1,
        label='Рабочая зона'
    )

    # Настройка осей и заголовка
    ax.set_xlabel('Угол визирования ϒ, градусы', fontsize=12)
    ax.set_ylabel('Расстояние R₀, км', fontsize=12)
    ax.set_title('Зависимость расстояния до спутника от угла визирования', fontsize=14)
    
    # Настройка сетки
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Лимиты осей
    ax.set_xlim(
        max(GAMMA_MIN_DEG, GAMMA_WORK_ZONE_MIN - 5), 
        min(GAMMA_PLOT_MAX, GAMMA_WORK_ZONE_MAX + 5)
    )
    ax.set_ylim(R_min * 0.95, R_max * 1.05)
    
    # Легенда
    ax.legend(
        loc='upper right', 
        frameon=True, 
        fontsize=10,
        title='Условные обозначения:'
    )

    # Сохранение графика
    if save_path:
        # Создание директории при необходимости
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Параметры сохранения
        fig.savefig(
            save_path,
            dpi=300,  # Высокое разрешение
            bbox_inches='tight',  # Обрезка пустых областей
            facecolor='#ffffff',  # Цвет фона при сохранении
            format='png'  # Формат файла
        )
        print(f"График успешно сохранен: {save_path}")
    
    # Отображение графика
    plt.tight_layout()
    plt.show()


def _test() -> None:
    """
    Тестовая функция для проверки работы модуля
    """
    try:
        # Загрузка TLE из файла для спутника с номером 56756
        s_name, tle_1, tle_2 = read_tle_base_file(DEFAULT_SATELLITE_ID)
        print(f"Загружен TLE для спутника: {s_name}")
        
        # Координаты наземного объекта (Санкт-Петербург)
        target_pos = DEFAULT_TARGET_POS

        # Настройка временного интервала
        start_time = datetime(2024, 2, 21, 3, 0, 0)
        time_delta = timedelta(seconds=10)  # Шаг расчета 10 секунд
        end_time = start_time + timedelta(days=16)  # Период 16 дней

        print(f"Расчет на период: {start_time} - {end_time}")
        print(f"Шаг расчета: {time_delta}")

        # Основной расчет
        R_0, gamma_grad = calculate_orbital_data(
            tle_1, 
            tle_2, 
            start_time, 
            end_time, 
            time_delta, 
            target_pos,
            GAMMA_PLOT_MAX
        )

        print(f"Получено {len(R_0)} точек данных")

        # Генерация имени файла с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"Печать/Picture_4_{timestamp}.png"
        
        # Визуализация и сохранение
        plot_orbital_data(R_0, gamma_grad, save_path)
        
    except Exception as e:
        print(f"Ошибка при выполнении теста: {e}")
        raise

if __name__ == "__main__":
    # Точка входа при запуске скрипта
    _test()