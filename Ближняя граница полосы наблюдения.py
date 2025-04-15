import numpy as np
import matplotlib.pyplot as plt

# Параметры
H = 570  # Высота орбиты [км]
theta = np.linspace(24, 55, 100)  # Углы визирования [град]
theta_rad = np.deg2rad(theta)

# Наклонная дальность
R_min = H / np.cos(theta_rad)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(theta, R_min, 'b-', linewidth=2)
plt.title('Ближняя граница полосы наблюдения', fontsize=14)
plt.xlabel('Угол визирования, град', fontsize=12)
plt.ylabel('Наклонная дальность, км', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(500, 1000)
plt.tight_layout()
plt.show()