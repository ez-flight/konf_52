import numpy as np
import matplotlib.pyplot as plt

# Параметры
height = 520e3  # высота орбиты в метрах
wavelength = 0.094  # длина волны в метрах
antenna_size = 5.0  # размер антенны в метрах

# Углы визирования
angles = np.linspace(20, 60, 100)  # от 0 до 45 градусов
angles_rad = np.radians(angles)

# Расчет расстояния до цели
distance = height / np.cos(angles_rad)

# Расчет ширины полосы обзора
swath_width = 2 * distance * np.tan(np.arcsin((wavelength / antenna_size) / 2))/1000

# Построение графика
plt.plot(angles, swath_width)
plt.fill_between(angles, 10, swath_width, color='lightgreen', alpha=0.3, label='Рабочая зона')
plt.title('Зависимость полосы обзора от угла визирования')
plt.xlabel('Угол визирования (градусы)')
plt.ylabel('Ширина полосы обзора (км)')
plt.grid()
# Добавление меток для углов 24 и 55
plt.axvline(24, color='green', linestyle=':', linewidth=2, label=f'$\\theta_{{min}}$ = 24° (КА Кондор ФКА)')    # Новая линия
plt.axvline(55, color='purple', linestyle=':', linewidth=2, label=f'$\\theta_{{max}}$ = 55° (КА Кондор ФКА)')   # Новая линия
plt.savefig('2_Prob/2_work_zone.png', dpi=300)
plt.show()