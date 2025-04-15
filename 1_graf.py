import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton
import os

# Создаем папку для результатов
os.makedirs('result', exist_ok=True)

# Параметры системы для S-диапазона
R_earth = 6371e3  # Радиус Земли (м)
H = 520e3         # Высота орбиты (м)
lambda_ = 0.094   # Длина волны РСА (9.4 см для S-диапазона)
D_ant = 10.0      # Размер антенны (м)

# Расчет ширины луча АФАР
theta_beam = np.rad2deg(lambda_ / D_ant)  # Угол в градусах

def calculate_phi(B_deg):
    """Расчет центрального угла через уравнение геометрии"""
    B_rad = np.deg2rad(B_deg)
    def equation(phi):
        return (R_earth + H) * np.sin(B_rad) - R_earth * np.sin(B_rad + phi)
    
    try:
        return newton(equation, x0=0.5, maxiter=100, tol=1e-6)
    except RuntimeError:
        return np.nan

# Расчет полосы обзора для диапазона углов
theta_angles = np.linspace(24, 55, 100)
swath_widths = []

for theta in theta_angles:
    phi = calculate_phi(theta)
    if not np.isnan(phi):
        swath_km = R_earth * phi / 1000
        swath_widths.append(swath_km)
    else:
        swath_widths.append(0)

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(theta_angles, swath_widths, 'darkorange', linewidth=2)
plt.title('Зависимость ширины полосы обзора от угла визирования\n(S-диапазон, λ=9.4 см)', 
          fontsize=14, pad=20)
plt.xlabel('Угол визирования [град]', fontsize=12)
plt.ylabel('Ширина полосы [км]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)



plt.tight_layout()
# Сохранение в файл
plt.savefig('result/1_swath_s-band.png', 
           dpi=300, 
           bbox_inches='tight')
plt.show()

print(f"Ширина луча антенны: {theta_beam:.2f}°")