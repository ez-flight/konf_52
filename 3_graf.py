import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton
import os

# Создаем папку для результатов
os.makedirs('result', exist_ok=True)

# Параметры системы для S-диапазона
R_earth = 6371e3      # Радиус Земли (м)
H = 500e3             # Высота орбиты (м)
c = 3e8               # Скорость света (м/с)
lambda_ = 0.094       # Длина волны РСА (9.4 см)
D_ant = 10.0          # Размер антенны (м)
orbital_speed = 7600  # Орбитальная скорость (м/с)

# Расчет азимутального разрешения и PRF ограничения
delta_az = (lambda_ * H) / (2 * D_ant)        # Азимутальное разрешение (м)
prf_azimuth = 2 * orbital_speed / delta_az    # Минимальная PRF по азимуту (Гц)

def calculate_phi(B_deg):
    """Расчет центрального угла через уравнение геометрии"""
    B_rad = np.deg2rad(B_deg)
    def equation(phi):
        # Исправленная строка: B_earth → theta_rad + phi
        return (R_earth + H) * np.sin(B_rad) - R_earth * np.sin(B_rad + phi)
    
    try:
        return newton(equation, x0=0.5, maxiter=100, tol=1e-6)
    except RuntimeError:
        return np.nan

def calculate_prf_range(B_deg):
    """Расчет максимальной PRF по дальности"""
    phi = calculate_phi(B_deg)
    if np.isnan(phi):
        return 0
    
    # Расчет наклонной дальности
    S = np.sqrt((R_earth + H)**2 + R_earth**2 
                - 2*R_earth*(R_earth + H)*np.cos(phi))
    return c / (2 * S)  # В Гц

# Диапазон углов визирования
theta_angles = np.linspace(24, 55, 100)
prf_range = [calculate_prf_range(theta) for theta in theta_angles]
prf_azimuth_list = [prf_azimuth] * len(theta_angles)  # Горизонтальная линия

# Визуализация
plt.figure(figsize=(12, 6))

# Отрисовка обоих ограничений
plt.plot(theta_angles, prf_range, 'royalblue', linewidth=2, label='По дальности')
plt.plot(theta_angles, prf_azimuth_list, 'forestgreen', linestyle='--', 
         linewidth=2, label='По азимуту')

# Область допустимых значений
plt.fill_between(theta_angles, prf_azimuth_list, prf_range, 
                 where=(np.array(prf_range) > prf_azimuth),
                 color='lightgreen', alpha=0.3, label='Рабочая зона')

plt.title('Ограничения PRF для РСА\n(S-диапазон, λ=9.4 см, H=500 км)', 
          fontsize=14, pad=20)
plt.xlabel('Угол визирования [град]', fontsize=12)
plt.ylabel('PRF [Гц]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 300)  # Фиксированный диапазон для наглядности
plt.legend()



plt.tight_layout()
# Сохранение в файл
plt.savefig('result/3_prf_constraints.png', 
           dpi=300, 
           bbox_inches='tight')
plt.show()

# Вывод параметров
print(f"Азимутальное разрешение: {delta_az:.2f} м")
print(f"Минимальная PRF по азимуту: {prf_azimuth:.0f} Гц")