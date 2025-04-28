import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Параметры системы
R_earth = 6371e3       # Радиус Земли [м]
H = 500e3              # Высота орбиты [м]
theta_min = 24         # Минимальный угол визирования [°]
theta_max = 55         # Максимальный угол визирования [°]
tau = 10e-6            # Длительность импульса [с]
c = 3e8                # Скорость света [м/с]

# Функция связи углов θ и γ
def equation(theta_deg, gamma):
    theta = np.radians(theta_deg)
    return np.sin(theta + gamma) - (R_earth / (R_earth + H)) * np.sin(theta)

# Находим γ для theta_min и theta_max
gamma_min = fsolve(lambda gamma: equation(theta_min, gamma), 0.1)[0]
gamma_max_theta = fsolve(lambda gamma: equation(theta_max, gamma), 0.1)[0]

# Расчёт наклонных дальностей для границ (геометрических)
R_min_geometry = np.sqrt((R_earth + H)**2 + R_earth**2 - 2 * R_earth * (R_earth + H) * np.cos(gamma_min)) / 1000  # [км]
R_max_geometry = np.sqrt((R_earth + H)**2 + R_earth**2 - 2 * R_earth * (R_earth + H) * np.cos(gamma_max_theta)) / 1000  # [км]

# Минимальная дальность из-за задержки сигнала
R_min_delay = (c * tau) / 2  # В метрах
R_min_delay_km = R_min_delay / 1000  # Переводим в километры

# Выбираем максимальное значение R_min
R_min = max(R_min_geometry, R_min_delay_km)

# Генерация кривой θ(R)
gamma_values = np.linspace(gamma_min, gamma_max_theta, 100)
theta_values = []
R_values = []

for gamma in gamma_values:
    theta = fsolve(lambda theta: equation(theta, gamma), theta_min)[0]
    theta_values.append(theta)
    R = np.sqrt((R_earth + H)**2 + R_earth**2 - 2 * R_earth * (R_earth + H) * np.cos(gamma)) / 1000
    R_values.append(R)


# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(R_values, theta_values, 'b-', label='Зависимость θ(R)')
plt.axvline(R_min, color='r', linestyle='-.', label=f'R_min = {R_min:.1f} км')
plt.axvline(R_max_geometry, color='r', linestyle='--', label=f'R_max = {R_max_geometry:.1f} км')
plt.axhline(theta_min, color='b', linestyle='-.', label=f'g_min = {theta_min:.1f} км')
plt.axhline(theta_max, color='b', linestyle='--', label=f'g_max = {theta_max:.1f} км')

# Заливка рабочей зоны (между theta_min и кривой θ(R))
plt.fill_between(R_values, theta_min, theta_values, color='green', alpha=0.2, label='Рабочая зона')

# Если R_min_delay доминирует, добавляем аннотацию
if R_min_delay_km > R_min_geometry:
    plt.axvline(R_min_delay_km, color='orange', linestyle=':', label=f'R_min_delay = {R_min_delay_km:.1f} км (импульс)')

plt.xlabel('Наклонная дальность, R (км)', fontsize=12)
plt.ylabel('Угол визирования, g (°)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.9))
plt.savefig('Печать/Picture_4.png', dpi=300)
plt.show()

# Вывод информации о границах
print(f"Геометрическая R_min: {R_min_geometry:.1f} км")
print(f"R_min из-за задержки: {R_min_delay_km:.1f} км")
print(f"Итоговая R_min: {R_min:.1f} км")