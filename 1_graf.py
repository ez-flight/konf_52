import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import newton

# Параметры системы
R_earth = 6371e3  # Радиус Земли (м)
H = 500e3         # Высота орбиты (м)
lambda_ = 0.05    # Длина волны РСА (м, пример для C-диапазона)
D_ant = 10.0      # Размер антенны (м)

# Расчет ширины луча АФАР
theta_beam = np.rad2deg(lambda_ / D_ant)  # Угол в градусах

# Углы визирования (B_min и B_max)
B_min = 20  # Минимальный угол (град)
B_max = 50  # Максимальный угол (град)

# Функция для расчета центрального угла phi
def calculate_phi(B_deg):
    B_rad = np.deg2rad(B_deg)
    def equation(phi):
        return (R_earth + H) * np.sin(B_rad) - R_earth * np.sin(B_rad + phi)
    try:
        phi = newton(equation, x0=0.1, maxiter=100, tol=1e-6)
        return phi
    except RuntimeError:
        return np.nan

# Расчет полосы обзора
phi_min = calculate_phi(B_min)
phi_max = calculate_phi(B_max)
delta_L = R_earth * (phi_max - phi_min) / 1e3  # В км

# Вывод результатов
print(f"Ширина луча АФАР: {theta_beam:.2f}°")
print(f"Полоса обзора ΔL: {delta_L:.1f} км")

B_angles = np.linspace(24, 55, 50)
delta_L_values = []
for B in B_angles:
    phi = calculate_phi(B)
    if not np.isnan(phi):
        delta_L_values.append(R_earth * phi / 1e3)
    else:
        delta_L_values.append(0)

plt.figure(figsize=(10, 6))
plt.plot(B_angles, delta_L_values, 'b-', linewidth=2)
plt.title('Полоса обзора ΔL vs. Угол визирования B')
plt.xlabel('Угол визирования B, градусы')
plt.ylabel('ΔL, км')
plt.grid(True)
plt.show()