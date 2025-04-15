import numpy as np
import matplotlib.pyplot as plt

# Параметры системы
H = 570e3          # Высота орбиты [м]
c = 3e8            # Скорость света [м/с]
theta = np.linspace(20, 60, 100)  # Углы визирования [град]
lambda_ = 0.1      # Длина волны S-диапазона [м]
v = 7600           # Орбитальная скорость [м/с]

# Расчет наклонной дальности
theta_rad = np.deg2rad(theta)
R = H / np.cos(theta_rad)
R_min, R_max = R.min(), R.max()

# Расчет PRF_min и PRF_max (по R_max и R_min)
PRF_min = c / (2 * R_max)  # Минимальная PRF (временное ограничение)
PRF_max = c / (2 * R_min)  # Максимальная PRF (доплеровское ограничение)

# Формирование данных
R_vals = np.linspace(R_min, R_max, 100)
PRF_vals = c / (2 * R_vals)  # Теоретическая зависимость PRF(R)

# Построение графика
plt.figure(figsize=(12, 7))

# Теоретическая кривая PRF
plt.plot(R_vals / 1e3, PRF_vals / 1e3, 'k-', linewidth=2, label='PRF(R) = c/(2R)')

# Горизонтальные ограничения
plt.axhline(PRF_min / 1e3, color='r', linestyle='--', label=f'PRF_min = {PRF_min/1e3:.1f} кГц')
plt.axhline(PRF_max / 1e3, color='b', linestyle='--', label=f'PRF_max = {PRF_max/1e3:.1f} кГц')

# Рабочая зона
plt.fill_between(
    R_vals / 1e3, 
    PRF_min / 1e3, 
    PRF_max / 1e3, 
    where=(PRF_vals >= PRF_min) & (PRF_vals <= PRF_max),
    color='green', 
    alpha=0.2, 
    label='Рабочая зона'
)

# Настройки графика
plt.title('Зависимость PRF от наклонной дальности (S-диапазон, КА "Кондор") в режиме ScanSAR', fontsize=14, pad=15)
plt.xlabel('Наклонная дальность, км', fontsize=12)
plt.ylabel('PRF, кГц', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('good/PRF_vs_Range_Кондор.png', dpi=300, bbox_inches='tight')
plt.show()