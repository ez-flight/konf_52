import numpy as np
import matplotlib.pyplot as plt

# Константы
R_earth = 6371e3  # Радиус Земли в метрах
mu = 3.986e14     # Геоцентрическая гравитационная постоянная (м³/с²)
theta_deg = 60    # Угол обзора в градусах

# Диапазон высот орбиты (в метрах)
H = np.linspace(300e3, 2000e3, 100)  # От 300 км до 2000 км

# Расчет параметров
a = R_earth + H                      # Большая полуось
T = 2 * np.pi * np.sqrt(a**3 / mu)   # Период обращения
T_contact = (theta_deg / 360) * T    # Длительность видеоконтакта

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(H/1e3, T_contact/60, 'purple', linewidth=2)  # Переводим в км и минуты
plt.title('Зависимость длительности видеоконтакта от высоты орбиты\n(θ = 60°)', fontsize=14)
plt.xlabel('Высота орбиты, км', fontsize=12)
plt.ylabel('Длительность видеоконтакта, мин', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Добавление формулы
plt.text(1200, 40, 
         r'$T_{конт} = \frac{\theta}{360} \cdot 2\pi \sqrt{\frac{(R_{Earth} + H)^3}{\mu}}$' + '\n' +
         r'$\theta=60^\circ$',
         fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('Prob/4_T_contact_vs_H.png', dpi=300)
plt.show()