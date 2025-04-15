import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_satellite_coverage(orbit_altitude=500e3, max_view_angle=60, earth_radius=6371e3):
    # Параметры сферы Земли
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Параметры орбиты спутника
    theta = np.linspace(0, 2*np.pi, 50)
    orbit_radius = earth_radius + orbit_altitude
    x_orbit = orbit_radius * np.cos(theta)
    y_orbit = orbit_radius * np.sin(theta)
    z_orbit = np.zeros_like(x_orbit)

    # Параметры зоны обзора
    cone_angle = np.deg2rad(max_view_angle)
    t = np.linspace(0, 2*np.pi, 30)
    h = np.linspace(0, orbit_radius*0.5, 2)
    t, h = np.meshgrid(t, h)

    # Создание 3D-графика
    fig = plt.figure(figsize=(16, 12))
    
    # Земля и орбита
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(x_earth, y_earth, z_earth, alpha=0.2, color='blue')
    ax.plot(x_orbit, y_orbit, z_orbit, '--', color='gray', alpha=0.5)
    
    # Зона обзора для разных положений
    for angle in np.linspace(0, np.pi/2, 3):
        x_cone = h * np.sin(cone_angle) * np.cos(t) + orbit_radius*np.cos(angle)
        y_cone = h * np.sin(cone_angle) * np.sin(t) + orbit_radius*np.sin(angle)
        z_cone = h * np.cos(cone_angle)
        ax.plot_surface(x_cone, y_cone, z_cone, alpha=0.3, color='orange')

    ax.set_title('3D модель зоны обзора РСА\n')
    ax.set_xlabel('X [м]'), ax.set_ylabel('Y [м]'), ax.set_zlabel('Z [м]')
    ax.set_box_aspect([1,1,1])

    # Вид сверху
    ax2 = fig.add_subplot(122)
    earth_circle = plt.Circle((0, 0), earth_radius, color='blue', alpha=0.2)
    orbit_circle = plt.Circle((0, 0), orbit_radius, color='gray', fill=False, linestyle='--')
    ax2.add_artist(earth_circle)
    ax2.add_artist(orbit_circle)
    
    # Проекция зоны обзора
    for angle in np.linspace(0, np.pi/2, 3):
        coverage_radius = orbit_altitude * np.tan(cone_angle)
        x = orbit_radius * np.cos(angle)
        y = orbit_radius * np.sin(angle)
        coverage = plt.Circle((x, y), coverage_radius, color='orange', alpha=0.3)
        ax2.add_artist(coverage)

    ax2.set_title('Проекция зоны обзора на плоскость орбиты\n')
    ax2.set_xlabel('X [м]'), ax2.set_ylabel('Y [м]')
    ax2.set_aspect('equal')
    ax2.set_xlim(-1.2*orbit_radius, 1.2*orbit_radius)
    ax2.set_ylim(-1.2*orbit_radius, 1.2*orbit_radius)

    plt.tight_layout()
    plt.savefig('result/3d_coverage.png', dpi=300)
    plt.show()

# Запуск визуализации для высоты 500 км и угла обзора 60°
plot_satellite_coverage(orbit_altitude=500e3, max_view_angle=60)