import numpy as np
import matplotlib.pyplot as plt

# Параметры системы
lambda_ = 0.094  # Длина волны (S-диапазон, 9.4 см)
H = 500e3        # Высота орбиты (500 км)
L_ant = 10.0     # Длина антенны (м)

# Расчет ширины полосы обзора
theta = np.linspace(20, 60, 100)  # Углы визирования в градусах
W = (lambda_ * H) / (L_ant * np.sin(np.deg2rad(theta))) / 1e3  # В км

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(theta, W, 'b-', linewidth=2)
plt.title('Зависимость ширины полосы обзора от угла визирования', fontsize=14)
plt.xlabel('Угол визирования, градусы', fontsize=12)
plt.ylabel('Ширина полосы обзора, км', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Prob/1_W_vs_theta.png', dpi=300)
plt.show()