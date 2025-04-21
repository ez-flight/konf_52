import numpy as np
import matplotlib.pyplot as plt

# Параметры системы
c = 3e8             # Скорость света [м/с]
angles = [24, 55]   # Углы визирования [град]

# Диапазон максимальных наклонных дальностей [м]
R_max = np.linspace(500e3, 2000e3, 100)  # От 500 км до 2000 км

# Построение графика
plt.figure(figsize=(12, 6))

# Для каждого угла визирования
for theta in angles:
    theta_rad = np.deg2rad(theta)
    
    # Расчет PRF
    PRF = c / (2 * R_max * np.sin(theta_rad))
    
    # Построение кривой
    plt.plot(R_max/1e3, PRF/1e3, 
             label=f'θ = {theta}°', 
             linewidth=2)

# Настройки графика
plt.title('Связь PRF и максимальной наклонной дальности', fontsize=14)
plt.xlabel('Максимальная наклонная дальность (R_max), км', fontsize=12)
plt.ylabel('Частота повторения импульсов (PRF), кГц', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Добавление формул
plt.text(1200, 120, 
         r'$PRF = \frac{c}{2 R_{max} \sin\theta}$',
         fontsize=14,
         bbox=dict(facecolor='white', alpha=0.8))

plt.legend()
plt.tight_layout()
plt.savefig('Prob/3_2_PRF_vs_Rmax_two_angles.png', dpi=300)
plt.show()