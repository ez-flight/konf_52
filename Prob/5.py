import numpy as np
import matplotlib.pyplot as plt

# Параметры системы
lambda_ = 0.094  # Длина волны (S-диапазон, 9.4 см)
H = 500e3        # Высота орбиты (500 км)
R_earth = 6371e3 # Радиус Земли (м)
mu = 3.986e14    # Геоцентрическая гравитационная постоянная (м³/с²)

# Расчет скорости КА для круговой орбиты
V = np.sqrt(mu / (R_earth + H))

# Диапазон углов рыскания (от -5° до 5°)
psi_deg = np.linspace(-5, 5, 100)
psi_rad = np.deg2rad(psi_deg)

# Расчет доплеровской частоты
f_d = (2 * V / lambda_) * np.sin(psi_rad)

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(psi_deg, f_d / 1e3, 'm-', linewidth=2)  # Переводим в кГц
plt.title('Доплеровская частота в зависимости от угла рыскания', fontsize=14)
plt.xlabel('Угол рыскания (ψ), градусы', fontsize=12)
plt.ylabel('Доплеровская частота (f_d), кГц', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=1)

# Добавление формулы и параметров
plt.text(-2, 2, 
         r'$f_d = \frac{2V}{\lambda} \sin\psi$' + '\n' +
         f'V = {V/1e3:.1f} км/с\n' +
         r'$\lambda = 0.094$ м',
         fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('Prob/5_f_d_vs_psi.png', dpi=300)
plt.show()