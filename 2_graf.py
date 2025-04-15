import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton
import os

# Создаем папку для результатов
os.makedirs('result', exist_ok=True)


# Параметры системы для S-диапазона
R_earth = 6371e3    # Радиус Земли (м)
H = 520e3           # Высота орбиты (м)
c = 3e8             # Скорость света (м/с)
lambda_ = 0.094     # Длина волны РСА (9.4 см)

def calculate_phi(B_deg):
    """Расчет центрального угла через уравнение геометрии"""
    B_rad = np.deg2rad(B_deg)
    def equation(phi):
        return (R_earth + H) * np.sin(B_rad) - R_earth * np.sin(B_rad + phi)
    
    try:
        return newton(equation, x0=0.5, maxiter=100, tol=1e-6)
    except RuntimeError:
        return np.nan

def calculate_prf(B_deg):
    """Расчет максимальной PRF для угла визирования"""
    phi = calculate_phi(B_deg)
    if np.isnan(phi):
        return 0
    
    # Расчет наклонной дальности
    S = np.sqrt((R_earth + H)**2 + R_earth**2 
                - 2*R_earth*(R_earth + H)*np.cos(phi))
    
    # Ограничение PRF по дальности
    return c / (2 * S)  # В Гц
    # Ограничение по азимуту
    
# Диапазон углов визирования
theta_angles = np.linspace(24, 55, 100)
prf_values = [calculate_prf(theta) for theta in theta_angles]

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(theta_angles, prf_values, 'royalblue', linewidth=2)
plt.title('Зависимость частоты повторения импульсов (PRF) от угла визирования\n(S-диапазон, λ=9.4 см)', 
          fontsize=14, pad=20)
plt.xlabel('Угол визирования [град]', fontsize=12)
plt.ylabel('PRF [Гц]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
# Сохранение в файл
plt.savefig('result/2_prf_vs_angle.png', 
           dpi=300, 
           bbox_inches='tight')
plt.show()