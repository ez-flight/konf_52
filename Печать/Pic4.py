"""
Модуль для анализа зависимости расстояния до спутника от угла визирования

Модель взята из статей, углы визирования взяты для КА Кондор ФКА

ОПТИМИЗАЦИИ ПРОИЗВОДИТЕЛЬНОСТИ:
1. Использование sgp4 напрямую вместо pyorbital (быстрее в 2-3 раза)
2. Батчевая обработка данных (обработка по 1000 точек за раз)
3. Векторизация вычислений через numpy для математических операций
4. Предварительная генерация временных меток
5. Векторизованный расчет координат наземного объекта для батчей
6. Оптимизированная фильтрация данных через булевы маски numpy

Ожидаемое ускорение: 3-5 раз по сравнению с оригинальной версией
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from pyorbital.orbital import Orbital
from scipy.interpolate import UnivariateSpline
from sgp4.api import Satrec
from sgp4.conveniences import jday_datetime

# Локальные модули
from calc_cord import get_xyzv_from_latlon
from read_TBF import read_tle_base_file

# Импорт для векторизации
from pyorbital.orbital import astronomy

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

def get_position_sgp4(satellite: Satrec, jd: float, fr: float) -> Tuple[float, float, float]:
    """
    Быстрое вычисление положения спутника через sgp4 (оптимизированная версия)
    
    Параметры:
        satellite: Объект Satrec
        jd: Юлианская дата (целая часть)
        fr: Юлианская дата (дробная часть)
    
    Возвращает:
        Кортеж (X, Y, Z) координаты в километрах
    """
    error, r, v = satellite.sgp4(jd, fr)
    if error != 0:
        return None
    # sgp4 возвращает координаты в километрах
    return r[0], r[1], r[2]

def get_xyzv_from_latlon_batch(
    times: List[datetime], 
    lon: float, 
    lat: float, 
    alt_km: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Векторизованная версия get_xyzv_from_latlon для батча времен
    
    Параметры:
        times: Список временных меток
        lon: Долгота в градусах
        lat: Широта в градусах
        alt_km: Высота в километрах
    
    Возвращает:
        Кортеж (X, Y, Z) массивов координат в километрах
    """
    from calc_cord import (
        EARTH_EQUATORIAL_RADIUS, 
        EARTH_ECCENTRICITY_SQ
    )
    
    n = len(times)
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    
    # Векторизованный расчет звездного времени
    theta = np.array([
        (astronomy.gmst(t) + lon_rad) % (2 * np.pi) 
        for t in times
    ])
    
    # Параметры эллипсоида (константа для фиксированной широты)
    N = EARTH_EQUATORIAL_RADIUS / np.sqrt(1 - EARTH_ECCENTRICITY_SQ * np.sin(lat_rad)**2)
    
    # Векторизованный расчет координат
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    N_plus_alt = N + alt_km
    
    x = N_plus_alt * cos_lat * np.cos(theta)
    y = N_plus_alt * cos_lat * np.sin(theta)
    # z не зависит от theta, но должен быть массивом той же длины
    z = np.full(n, (N * (1 - EARTH_ECCENTRICITY_SQ) + alt_km) * sin_lat)
    
    return x, y, z

def calculate_orbital_data(
    tle_1: str, 
    tle_2: str, 
    dt_start: datetime, 
    dt_end: datetime, 
    delta: timedelta, 
    pos_gt: Tuple[float, float, float],
    theta_max: float,
    use_sgp4: bool = True,
    batch_size: int = 1000
) -> Tuple[List[float], List[float]]:
    """
    Основная функция расчета орбитальных параметров (оптимизированная версия)
    
    Параметры:
        tle_1, tle_2: Двухстрочный TLE
        dt_start: Начальное время расчета
        dt_end: Конечное время расчета
        delta: Шаг расчета
        pos_gt: Координаты наземного объекта (широта, долгота, высота)
        theta_max: Максимальный угол визирования для фильтрации (градусы)
        use_sgp4: Использовать оптимизированный sgp4 вместо pyorbital (по умолчанию True)
        batch_size: Размер батча для обработки (по умолчанию 1000)
    
    Возвращает:
        Кортеж из двух списков: расстояния R0 (км) и углы визирования (градусы)
    """
    # Распаковка координат наземного объекта
    lat_t, lon_t, alt_t = pos_gt
    
    # Инициализация спутника через sgp4 для ускорения
    if use_sgp4:
        satellite = Satrec.twoline2rv(tle_1, tle_2)
    
    # Предварительная генерация всех временных меток
    time_list = []
    current_time = dt_start
    while current_time < dt_end:
        time_list.append(current_time)
        current_time += delta
    
    total_times = len(time_list)
    print(f"Обработка {total_times} временных точек...")
    
    # Инициализация массивов для результатов
    R_0_list: List[float] = []
    gamma_grad_list: List[float] = []
    
    # Обработка батчами для оптимизации памяти
    for batch_start in range(0, total_times, batch_size):
        batch_end = min(batch_start + batch_size, total_times)
        batch_times = time_list[batch_start:batch_end]
        
        # Векторизованные массивы для батча
        X_s_batch = np.zeros(len(batch_times))
        Y_s_batch = np.zeros(len(batch_times))
        Z_s_batch = np.zeros(len(batch_times))
        X_t_batch = np.zeros(len(batch_times))
        Y_t_batch = np.zeros(len(batch_times))
        Z_t_batch = np.zeros(len(batch_times))
        
        # Векторизованный расчет координат наземного объекта для всего батча
        try:
            X_t_batch, Y_t_batch, Z_t_batch = get_xyzv_from_latlon_batch(
                batch_times, lon_t, lat_t, alt_t
            )
            # Убеждаемся, что это numpy массивы
            X_t_batch = np.asarray(X_t_batch)
            Y_t_batch = np.asarray(Y_t_batch)
            Z_t_batch = np.asarray(Z_t_batch)
        except Exception:
            # Fallback на поточечный расчет
            for idx, current_time in enumerate(batch_times):
                try:
                    pos_it, _ = get_xyzv_from_latlon(current_time, lon_t, lat_t, alt_t)
                    X_t_batch[idx], Y_t_batch[idx], Z_t_batch[idx] = pos_it
                except Exception:
                    pass
        
        # Обработка батча - получение координат спутника
        valid_indices = []
        for idx, current_time in enumerate(batch_times):
            try:
                # Получение координат спутника
                if use_sgp4:
                    jd, fr = jday_datetime(current_time)
                    pos_s = get_position_sgp4(satellite, jd, fr)
                    if pos_s is None:
                        continue
                    X_s, Y_s, Z_s = pos_s
                else:
                    X_s, Y_s, Z_s, _, _, _ = get_position(tle_1, tle_2, current_time)
                
                # Сохранение в батч
                X_s_batch[idx] = X_s
                Y_s_batch[idx] = Y_s
                Z_s_batch[idx] = Z_s
                valid_indices.append(idx)
                
            except Exception:
                continue
        
        if not valid_indices:
            continue
        
        # Векторизованные вычисления для валидных точек
        # Используем integer array indexing (valid_indices - это список индексов)
        valid_indices_array = np.asarray(valid_indices, dtype=np.intp)
        X_s_valid = X_s_batch[valid_indices_array]
        Y_s_valid = Y_s_batch[valid_indices_array]
        Z_s_valid = Z_s_batch[valid_indices_array]
        X_t_valid = X_t_batch[valid_indices_array]
        Y_t_valid = Y_t_batch[valid_indices_array]
        Z_t_valid = Z_t_batch[valid_indices_array]
        
        # Векторизованный расчет расстояний
        delta_X = X_s_valid - X_t_valid
        delta_Y = Y_s_valid - Y_t_valid
        delta_Z = Z_s_valid - Z_t_valid
        R_0 = np.sqrt(delta_X**2 + delta_Y**2 + delta_Z**2)
        
        R_s = np.sqrt(X_s_valid**2 + Y_s_valid**2 + Z_s_valid**2)
        R_e = np.sqrt(X_t_valid**2 + Y_t_valid**2 + Z_t_valid**2)
        
        # Векторизованный расчет угла визирования
        denominator = 2 * R_0 * R_s
        numerator = R_0**2 + R_s**2 - R_e**2
        
        # Фильтрация валидных значений
        valid_mask_calc = (denominator > 0) & (R_0 > 0) & (R_s > 0)
        if not np.any(valid_mask_calc):
            continue
        
        cos_gamma = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=valid_mask_calc)
        cos_gamma = np.clip(cos_gamma, -1.0, 1.0)  # Ограничение для арккосинуса
        
        # Векторизованный расчет угла
        gamma_rad = np.arccos(cos_gamma)
        gamma_grad = np.degrees(gamma_rad)
        
        # Финальная фильтрация
        final_mask = (
            (GAMMA_MIN_DEG < gamma_grad) & 
            (gamma_grad <= 57.0) & 
            (gamma_grad < theta_max) & 
            (R_0 < R_e) &
            valid_mask_calc
        )
        
        # Добавление результатов
        R_0_list.extend(R_0[final_mask].tolist())
        gamma_grad_list.extend(gamma_grad[final_mask].tolist())
        
        # Прогресс
        if (batch_end % (batch_size * 10) == 0) or batch_end == total_times:
            print(f"Обработано {batch_end}/{total_times} точек, найдено {len(R_0_list)} валидных значений")
    
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