#График 5.10
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

# Параметры системы
lambda_ = 0.094  # Длина волны (S-диапазон, 9.4 см)
H = 520e3        # Высота орбиты (500 км)
R_earth = 6378e3 # Радиус Земли (м)
mu = 3.986e14    # Геоцентрическая гравитационная постоянная (м³/с²)

# Расчет скорости КА для круговой орбиты
V = np.sqrt(mu / (R_earth + H))

# Диапазон углов рыскания (от -5° до 5°)
psi_deg = np.linspace(-5, 5, 100)
psi_rad = np.deg2rad(psi_deg)

# Расчет доплеровской частоты
f_d = (2 * V / lambda_) * np.sin(psi_rad)

# Генерация имени файла с временной меткой
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"Печать/Picture_3_{timestamp}.png"

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(psi_deg, f_d / 1e3, 'm-', linewidth=2)  # Переводим в кГц
#plt.title('Доплеровская частота в зависимости от угла рыскания', fontsize=14)
plt.xlabel('Угол рыскания, градусы', fontsize=12)
plt.ylabel('Доплеровская частота, кГц', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=1)

# Добавление формулы и параметров
plt.text(2, 2, 
         f'V = {V/1e3:.1f} км/с\n' +
         r'$\lambda = 0.094$ м',
         fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()