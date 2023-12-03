import numpy as np

np.random.seed(0)

# Генерация случайных точек вокруг центров кластеризации
cluster1_center = [2, 2]
cluster1_points = np.random.randn(20, 2) + cluster1_center

cluster2_center = [-2, -2]
cluster2_points = np.random.randn(20, 2) + cluster2_center

# Координаты всех точек
points = np.concatenate([cluster1_points, cluster2_points])

# Создание сети Кохонена с заданными входами и количеством нейронов
num_inputs = 2
num_neurons = 2
weights = np.random.randn(num_neurons, num_inputs)

# Обучение сети путем последовательного предъявления точек
learning_rate = 0.01
num_epochs = 50

for epoch in range(num_epochs):
    for point in points:
        # Распространение сигнала через сеть
        distances = np.linalg.norm(point - weights, axis=1)
        winner_neuron = np.argmin(distances)

        # Обновление весов выигравшего нейрона
        weights[winner_neuron] += learning_rate * (point - weights[winner_neuron])

    # Уменьшение коэффициента выбора
    learning_rate = (50 - epoch) / 100

# Проверка принадлежности точек к кластерам
for point in points:
    distances = np.linalg.norm(point - weights, axis=1)
    winner_neuron = np.argmin(distances)
    cluster_label = "Cluster 1" if winner_neuron == 0 else "Cluster 2"
    print(f"Point {point} belongs to {cluster_label}")
