import numpy as np
import matplotlib.pyplot as plt

Rz = 6371  # Радиус Земли в км
H = np.linspace(200, 2000, 100)  # Диапазон высот: 200–2000 км

theta_crit = np.rad2deg(np.arcsin(Rz / (Rz + H)))

plt.figure(figsize=(10, 6))
plt.plot(H, theta_crit, 'r-', linewidth=2)
plt.title('Зависимость критического угла визирования от высоты орбиты', fontsize=14)
plt.xlabel('Высота орбиты (H), км', fontsize=12)
plt.ylabel('Критический угол ($\\theta_{\\text{крит}}$), градусы', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()