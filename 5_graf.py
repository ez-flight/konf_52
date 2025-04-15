import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
import seaborn as sns

# Инициализация стилей
sns.set_theme(style="whitegrid")
os.makedirs('result', exist_ok=True)

# Тестовые данные
schedule_data = [
    {'start': '2024-02-20 08:00', 'duration': 15, 'target': 'A', 'type': 'Съемка'},
    {'start': '2024-02-20 08:30', 'duration': 25, 'target': 'B', 'type': 'Калибровка'},
    {'start': '2024-02-20 09:15', 'duration': 30, 'target': 'C', 'type': 'Съемка'},
    {'start': '2024-02-20 10:00', 'duration': 20, 'target': 'D', 'type': 'Мониторинг'},
]

# Цветовая схема должна быть объявлена ДО использования
colors = {
    'Съемка': '#1f77b4',
    'Калибровка': '#2ca02c', 
    'Мониторинг': '#ff7f0e'
}

# Преобразование данных
df = pd.DataFrame(schedule_data)
df['start'] = pd.to_datetime(df['start'])
df['end'] = df['start'] + pd.to_timedelta(df['duration'], unit='m')

# Конвертация в числовой формат
df['start_num'] = mdates.date2num(df['start'])
df['end_num'] = mdates.date2num(df['end'])
df['duration_num'] = df['end_num'] - df['start_num']

# Сбор временных меток
all_times = pd.concat([df['start'], df['end']]).unique()
time_ticks = mdates.date2num(np.sort(all_times))

# Построение графика
fig, ax = plt.subplots(figsize=(14, 7))

# Отрисовка полос
for idx, row in df.iterrows():
    ax.barh(
        y=row['target'],
        width=row['duration_num'],
        left=row['start_num'],
        color=colors[row['type']],
        edgecolor='black',
        height=0.6,
        label=row['type']
    )

# Форматирование временной оси
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45, ha='right')

# Вертикальные линии
for t in all_times:
    ax.axvline(x=mdates.date2num(t), 
               color='gray', 
               linestyle='--', 
               alpha=0.4,
               linewidth=0.8)

# Настройка подписей
ax.set_xlabel('Время съемки', fontsize=12, labelpad=10)
ax.set_ylabel('Целевые участки', fontsize=12, labelpad=10)
ax.set_title('Детальное расписание работы РСА\n20 февраля 2024 года', 
            fontsize=14, 
            pad=20)

# Легенда
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), 
          by_label.keys(),
          title='Тип операции',
          loc='upper right')

# Сохранение и вывод
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('result/5_schedule.png', dpi=300, bbox_inches='tight')
print(f'График сохранен: {os.path.abspath("result/5_schedule.png")}')
plt.show()