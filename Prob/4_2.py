import numpy as np
import matplotlib.pyplot as plt

# Константы
R_earth = 6371e3  # Радиус Земли в метрах
mu = 3.986e14     # Геоцентрическая гравитационная постоянная (м³/с²)
theta_degrees = [24, 55]  # Углы обзора в градусах

# Диапазон высот орбиты (в метрах)
H = np.linspace(300e3, 2000e3, 100)  # От 300 км до 2000 км

# Построение графика
plt.figure(figsize=(12, 6))

for theta_deg in theta_degrees:
    # Расчет параметров
    a = R_earth + H                      # Большая полуось
    T = 2 * np.pi * np.sqrt(a**3 / mu)   # Период обращения
    T_contact = (theta_deg / 360) * T    # Длительность видеоконтакта

    # Построение графика для текущего угла
    plt.plot(H/1e3, T_contact/60, linewidth=2, label=f'θ = {theta_deg}°')  # Переводим в км и минуты

# Настройка графика
plt.title('Зависимость длительности видеоконтакта от высоты орбиты', fontsize=14)
plt.xlabel('Высота орбиты, км', fontsize=12)
plt.ylabel('Длительность видеоконтакта, мин', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Добавление формулы
plt.text(1200, 40, 
         r'$T_{конт} = \frac{\theta}{360} \cdot 2\pi \sqrt{\frac{(R_{Earth} + H)^3}{\mu}}$',
         fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('Prob/4_2_T_contact_vs_H.png', dpi=300)
plt.show()