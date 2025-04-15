import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Настройка шрифтов для русских подписей
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

os.makedirs('result', exist_ok=True)

# Параметры оптимизации
iterations = 100  # Количество итераций

# Генерация искусственных данных процесса оптимизации
np.random.seed(42)
optimization_history = {
    'Общее покрытие': np.cumsum(np.random.rand(iterations) * 1e6),
    'Эффективность времени': 1 - np.abs(np.sin(np.linspace(0, 4*np.pi, iterations))) * 0.3,
    'Использование ресурсов': np.clip(np.random.normal(0.7, 0.1, iterations), 0.5, 0.9),
    'Конфликты задач': np.exp(-0.05*np.arange(iterations)) + np.random.rand(iterations)*0.1
}

# Нормализация данных для визуализации
for key in optimization_history:
    optimization_history[key] = (optimization_history[key] - np.min(optimization_history[key])) / \
                              (np.max(optimization_history[key]) - np.min(optimization_history[key]))

# Создание графиков
plt.figure(figsize=(14, 10))

# Основные графики критериев
plt.subplot(2, 1, 1)
for criterion, values in optimization_history.items():
    plt.plot(values, label=criterion, linewidth=2, alpha=0.8)

plt.title('Динамика критериев оптимизации расписания съёмки', fontsize=14, pad=15)
plt.xlabel('Номер итерации', fontsize=12)
plt.ylabel('Нормализованное значение', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right', fontsize=10)

# График скорости изменения
plt.subplot(2, 1, 2)
for criterion, values in optimization_history.items():
    derivatives = np.diff(values)
    plt.plot(derivatives, label=criterion, alpha=0.8, linestyle='--')

plt.title('Скорость изменения критериев (первая производная)', fontsize=14, pad=15)
plt.xlabel('Номер итерации', fontsize=12)
plt.ylabel('Скорость изменения', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=1)
plt.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('result/3!_optimization_metrics.png', dpi=300, bbox_inches='tight')
plt.show()