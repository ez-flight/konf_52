import numpy as np
import matplotlib.pyplot as plt
import os

# Создаем папку для результатов
os.makedirs('good', exist_ok=True)

# Параметры системы
H = 500e3          # Высота орбиты [м]
lambda_ = 0.094    # Длина волны (S-диапазон) [м]
D_ant = 10.0       # Размер антенны [м]

# Диапазон углов визирования (в градусах)
theta_deg = np.linspace(20, 60, 100)
theta_rad = np.deg2rad(theta_deg)

# Расчет ширины полосы для разных режимов
def swath_stripmap(theta):
    return (lambda_ * H) / (D_ant * np.sin(theta))

def swath_spotlight(theta):
    return 0.5 * swath_stripmap(theta)  # Упрощенная модель

def swath_scansar(theta):
    return 2.5 * swath_stripmap(theta)  # Упрощенная модель

# Вычисление данных
W_stripmap = swath_stripmap(theta_rad) / 1e3  # В км
W_spotlight = swath_spotlight(theta_rad) / 1e3
W_scansar = swath_scansar(theta_rad) / 1e3

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(theta_deg, W_stripmap, 'b-', linewidth=2, label='Stripmap')
plt.plot(theta_deg, W_spotlight, 'r--', linewidth=2, label='Spotlight')
plt.plot(theta_deg, W_scansar, 'g-.', linewidth=2, label='ScanSAR')

plt.title('Зависимость ширины полосы обзора от угла визирования\n', fontsize=14)
plt.xlabel('Угол визирования, градусы', fontsize=12)
plt.ylabel('Ширина полосы обзора, км', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='upper right')

plt.xticks(np.arange(24, 55, 5))
plt.ylim(0, 40)
plt.tight_layout()
plt.savefig('good/swath_vs_angle.png', dpi=300, 
           bbox_inches='tight')
plt.show()