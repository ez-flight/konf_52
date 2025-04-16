import numpy as np
import matplotlib.pyplot as plt

# Параметры системы
c = 3e8             # Скорость света [м/с]
theta = 30          # Угол визирования [град]
theta_rad = np.deg2rad(theta)

# Диапазон максимальных наклонных дальностей [м]
R_max = np.linspace(500e3, 2000e3, 100)  # От 500 км до 2000 км

# Расчет PRF
PRF = c / (2 * R_max * np.sin(theta_rad))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(R_max/1e3, PRF/1e3, 'g-', linewidth=2)  # Переводим в км и кГц
plt.title('Связь частоты повторени зондирующего импульса (PRF)\nи максимальной наклонной дальности\n(θ = 30°)', fontsize=14)
plt.xlabel('Максимальная наклонная дальность (R_max), км', fontsize=12)
plt.ylabel('Частота повторения импульсов (PRF), кГц', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Добавление формулы
plt.text(1200, 50, 
         r'$PRF = \frac{c}{2 R_{max} \sin\theta}$' + '\n' +
         r'$\theta=30^\circ$',
         fontsize=14, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('Prob/3_PRF_vs_Rmax.png', dpi=300)
plt.show()