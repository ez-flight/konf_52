import numpy as np
import matplotlib.pyplot as plt

# Параметры системы
lambda_ = 0.094       # Длина волны (м)
H = 520e3             # Высота орбиты (м)
L_ant = 6             # Длина антенны (м)
delta_az_max = 50     # Макс. разрешение по азимуту (м)
Rz = 6371e3           # Радиус Земли (м)

theta_deg = 30        # Угол визирования (градусы)
phi_deg = 45          # Угол скольжения (градусы)
W_ant = 0.5           # Размер антенны по углу места (0.5 м)


def calculate_L(Rz, theta_deg, phi_deg):
    """
    Вычисляет расстояние L от надира до цели через угол визирования и угол скольжения.

    Параметры:
    - Rz (float): Радиус Земли (м).
    - theta_deg (float): Угол визирования (градусы).
    - phi_deg (float): Угол скольжения (градусы).

    Возвращает:
    - L (float): Расстояние (м).
    """
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    
    numerator = np.sin(theta) * np.sin(phi)
    denominator = np.sqrt(1 - np.sin(theta)**2 * np.cos(phi)**2)
    alpha = np.arcsin(numerator / denominator)
    
    L = Rz * alpha
    return L

def calculate_footprint(H, lambda_, L_ant, W_ant, theta_deg):
    """
    Рассчитывает размер следа луча АФАР (ΔX и ΔY).

    Параметры:
    - H (float): Высота платформы (м).
    - lambda_ (float): Длина волны (м).
    - L_ant (float): Размер антенны по азимуту (м).
    - W_ant (float): Размер антенны по углу места (м).
    - theta_deg (float): Угол визирования (градусы).

    Возвращает:
    - delta_Y (float): Размер следа по азимуту (м).
    - delta_X (float): Размер следа по дальности (м).
    """
    theta = np.deg2rad(theta_deg)
    
    delta_Y = (H * lambda_ / L_ant) / np.sin(theta)
    delta_X = (H * lambda_ / W_ant) / np.cos(theta)
    
    return delta_Y, delta_X

def L_obzora (L,theta_min,theta,delta_Y):
    delta_L = ((L*theta)-(L*theta_min)) + (((delta_Y*theta)+(delta_Y*theta_min))/2)
    return delta_L

def main ():
    # Расчет углов
    theta_min = np.rad2deg(np.arcsin(lambda_ / L_ant))  # Ширина луча антенны
    theta_max = np.rad2deg(np.arcsin(Rz / (Rz + H)))    # Макс. угол до горизонта

    # Диапазон углов визирования
    theta = np.linspace(theta_min, theta_max, 100)
    # Расчет
    L = calculate_L(Rz, theta, phi_deg)
    #print(f"L = {L / 1e3:.2f} км")  # L = 2450.12 км
    delta_Y, delta_X = calculate_footprint(H, lambda_, L_ant, W_ant, theta)
    #S0 = (lambda_ * H) / (L_ant * np.sin(np.deg2rad(theta))) / 1e3  # Полоса обзора (км)
    S0 = L_obzora(L,theta_min,theta,delta_Y)


    # Построение графика
    plt.figure(figsize=(12, 6))
    plt.fill_between(theta, 0, S0, color='lightgreen', alpha=0.3, label='Рабочая зона')
    plt.plot(theta, S0, 'b-', linewidth=2, label='Полоса обзора $S_0$')
    plt.axvline(theta_min, color='r', linestyle='--', label=f'$\\theta_{{min}} = {theta_min:.2f}^\\circ$')
    plt.axvline(theta_max, color='orange', linestyle='--', label=f'$\\theta_{{max}} = {theta_max:.1f}^\\circ$')
    plt.axvline(24, color='green', linestyle=':', linewidth=2, label=f'$\\theta_{{min}}$ = 24° (КА Кондор ФКА)')    # Новая линия
    plt.axvline(55, color='purple', linestyle=':', linewidth=2, label=f'$\\theta_{{max}}$ = 55° (КА Кондор ФКА)')   # Новая линия

    plt.title('Рабочая зона РСА', fontsize=14)
    plt.xlabel('Угол визирования ($\\theta$), градусы', fontsize=12)
    plt.ylabel('Полоса обзора ($S_0$), км', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('2_Prob/work_zone.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()