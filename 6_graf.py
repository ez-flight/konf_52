import numpy as np
import matplotlib.pyplot as plt
import os

# Создаем папку для результатов
os.makedirs('result', exist_ok=True)

# Параметры системы
R_earth = 6371e3      # Радиус Земли [м]
H = 500e3             # Высота орбиты [м]
c = 3e8               # Скорость света [м/с]
freq = 3.2e9          # Частота [Гц] (S-диапазон)
lambda_ = c/freq      # Длина волны [м]
Pt = 5000             # Мощность передатчика [Вт]
G = 35                # Коэффициент усиления антенны [дБ]
sigma = 10            # ЭПР цели [м²]
B = 50e6              # Ширина полосы [Гц]
T = 300               # Температура шума [К]
L = 2.0               # Потери в системе [раз]

# Константы
k = 1.38e-23          # Постоянная Больцмана

def calculate_snr(theta_deg):
    """Расчет SNR от угла визирования"""
    theta = np.deg2rad(theta_deg)
    R = H / np.cos(theta)  # Наклонная дальность
    snr = (Pt * 10**(G/10)**2 * lambda_**2 * sigma / 
          ((4*np.pi)**3 * R**4 * k * T * B * L))
    return 10*np.log10(snr)  # В дБ

def positioning_error(theta_deg, resolution):
    """Ошибка геопозиционирования"""
    theta = np.deg2rad(theta_deg)
    return resolution / np.tan(theta)  # В метрах

# Диапазон углов визирования
theta_angles = np.linspace(24, 55, 100)

# Расчет параметров
az_res = (lambda_ * H) / (2 * 10)  # Азимутальное разрешение (антенна 10 м)
snr_values = [calculate_snr(theta) for theta in theta_angles]
error_values = [positioning_error(theta, az_res) for theta in theta_angles]

# Построение графиков
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# График SNR
ax1.plot(theta_angles, snr_values, 'royalblue', linewidth=2)
ax1.set_title('Зависимость SNR от угла визирования', fontsize=14)
ax1.set_xlabel('Угол визирования [град]', fontsize=12)
ax1.set_ylabel('SNR [дБ]', fontsize=12)
ax1.grid(True, linestyle='--')

# График ошибки позиционирования
ax2.plot(theta_angles, error_values, 'crimson', linewidth=2)
ax2.set_title('Ошибка геопозиционирования от угла визирования', fontsize=14)
ax2.set_xlabel('Угол визирования [град]', fontsize=12)
ax2.set_ylabel('Ошибка, м', fontsize=12)
ax2.grid(True, linestyle='--')

plt.tight_layout()
# Сохранение в файл
plt.savefig('result/6_grafik.png', 
           dpi=300, 
           bbox_inches='tight')
plt.show()

# Вывод параметров
print(f"Азимутальное разрешение: {az_res:.2f} м")
print(f"SNR при θ=30°: {calculate_snr(30):.1f} дБ")
print(f"Ошибка при θ=30°: {positioning_error(30, az_res):.1f} м")