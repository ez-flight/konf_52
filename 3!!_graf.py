import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import gravitational_constant as G
from scipy.integrate import quad

# Параметры Земли
M_earth = 5.972e24  # Масса Земли [кг]
R_earth = 6371e3     # Радиус Земли [м]
mu = G * M_earth     # Гравитационный параметр [м³/с²]

# Функция для расчета периода обращения
def orbital_period(a):
    return 2 * np.pi * np.sqrt(a**3 / mu)

# Функция для расчета времени контакта (упрощенная модель)
def contact_time(H, theta=30, orbit_type='circular'):
    theta_rad = np.deg2rad(theta)
    
    if orbit_type == 'circular':
        a = R_earth + H
        v = np.sqrt(mu / a)
        T = orbital_period(a)
        # Упрощенная формула для времени контакта
        t_contact = (R_earth * np.sin(theta_rad)) / v
    elif orbit_type == 'elliptic':
        H_peri = 500e3  # Фиксированный перигей [м]
        H_apogee = H
        a = (2*R_earth + H_peri + H_apogee) / 2
        r_peri = R_earth + H_peri
        v_peri = np.sqrt(mu * (2/r_peri - 1/a))
        # Время контакта в перигее (максимальное)
        t_contact = (R_earth * np.sin(theta_rad)) / v_peri
    
    return t_contact / 60  # Переводим в минуты

# Диапазон высот для анализа
H_circular = np.linspace(500e3, 2000e3, 50)  # Круговая орбита [м]
H_elliptic = np.linspace(1000e3, 3000e3, 50) # Апогей эллиптической орбиты [м]

# Расчет данных
t_circular = [contact_time(H, orbit_type='circular') for H in H_circular]
t_elliptic = [contact_time(H, orbit_type='elliptic') for H in H_elliptic]

# Построение графика
plt.figure(figsize=(14, 8))

plt.plot(H_circular/1e3, t_circular, 'b-', linewidth=2, 
         label='Круговая орбита (θ=30°)')
plt.plot(H_elliptic/1e3, t_elliptic, 'r--', linewidth=2, 
         label='Эллиптическая орбита (H_peri=500 км)')

plt.title('Длительность видеоконтакта vs высота орбиты\n', fontsize=16)
plt.xlabel('Высота орбиты (апогей для эллиптической), км', fontsize=14)
plt.ylabel('Длительность контакта, мин', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='upper right')

plt.xticks(np.arange(500, 3500, 500))
plt.yticks(np.arange(0, 25, 2))
plt.tight_layout()

plt.savefig('contact_time_vs_altitude.png', dpi=300)
plt.show()