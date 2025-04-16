import numpy as np
import matplotlib.pyplot as plt

# Параметры системы
lambda_ = 0.094  # Длина волны (S-диапазон, 9.4 см)
H = 500e3        # Высота орбиты (500 км)
L_ant = 10.0     # Длина антенны (м)

# Диапазон углов крена (от 0 до 60 градусов)
gamma_deg = np.linspace(0, 60, 100)
gamma_rad = np.deg2rad(gamma_deg)

# Расчет пространственного разрешения по азимуту
delta_az = (lambda_ * H) / (2 * L_ant * np.cos(gamma_rad))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(gamma_deg, delta_az, 'r-', linewidth=2)
plt.title('Влияние угла крена на пространственное разрешение по азимуту', fontsize=14)
plt.xlabel('Угол крена, градусы', fontsize=12)
plt.ylabel('Пространственное разрешение (δ_az), м', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(2000, 5000)  # Ограничение для наглядности

# Добавление формулы
plt.text(30, 3500, r'$\delta_{az} = \frac{\lambda H}{2 L_{ant} \cos\gamma}$', 
         fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('Prob/2_delta_az_vs_gamma.png', dpi=300)
plt.show()