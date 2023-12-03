from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances_argmin_min, pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Вариант задания: "blobs"
n_samples = 150
random_state = 28
cluster_std = 2
centers = 7

# Генерация синтетических данных "blobs"
x, y = make_blobs(n_samples=n_samples, random_state=random_state, cluster_std=cluster_std, centers=centers)

# Визуализация сгенерированных данных без привязки к классам
plt.figure('Визуализация сгенерированных данных без привязки к классам')
plt.scatter(x[:, 0], x[:, 1], c='b')
plt.show()

# Создание подзаголовков для дендрограмм
methods = ['single', 'complete', 'ward']

# Создание и отрисовка дендрограмм для каждого способа
for i, method in enumerate(methods):
    plt.figure(f"Dendrogram ({method} linkage)", figsize=(10, 5))
    plt.title(f"Dendrogram ({method} linkage)")
    dend = shc.dendrogram(shc.linkage(x, method=method))
    plt.show()

Z = linkage(x, method='complete')
k = 3
labels = fcluster(Z, k, criterion='maxclust')
centroids = np.zeros((k, x.shape[1]))
for i in range(1, k + 1):
    centroids[i - 1, :] = np.mean(x[labels == i, :], axis=0)

# Предположим, что x - это матрица объектов (признаки), labels - вектор с метками кластеров, centroids - матрица
# центроидов

# создаем график
plt.figure(figsize=(8, 6))

# отображаем объекты выборки
for i in np.unique(labels):
    plt.scatter(x[labels == i, 0], x[labels == i, 1], label='Cluster ' + str(i))

# отображаем центроиды
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='x', label='Centroids')

# добавляем легенду
plt.legend()
plt.show()

# Рассчитаем среднюю сумму квадратов расстояний до центроида
intra_cluster_distances = np.zeros(k)
for i in range(1, k + 1):
    centroid = centroids[i - 1]
    cluster_points = x[labels == i, :]
    distances = distance.cdist(cluster_points, [centroid], 'euclidean')
    intra_cluster_distances[i - 1] = np.sum(distances ** 2) / distances.shape[0]

average_intra_cluster_distance = np.mean(intra_cluster_distances)

# Рассчитаем среднюю сумму внутрикластерных расстояний
inter_cluster_distances = np.zeros((k, k))
for i in range(1, k + 1):
    for j in range(1, k + 1):
        distance_matrix = distance.cdist(x[labels == i, :], x[labels == j, :], 'euclidean')
        inter_cluster_distances[i - 1, j - 1] = np.mean(distance_matrix)

average_inter_cluster_distance = np.mean(inter_cluster_distances)

# Рассчитаем среднюю сумму межкластерных расстояний
within_cluster_distances = np.zeros(k)
for i in range(1, k + 1):
    for j in range(1, k + 1):
        if i != j:
            distance_matrix = distance.cdist(x[labels == i, :], x[labels == j, :], 'euclidean')
            within_cluster_distances[i - 1] += np.sum(distance_matrix) / (
                    distance_matrix.shape[0] * distance_matrix.shape[1])

average_within_cluster_distance = np.mean(within_cluster_distances)

# Выведем результаты
print("Средняя сумма квадратов расстояний до центроида:", average_intra_cluster_distance)
print("Средняя сумма внутрикластерных расстояний:", average_within_cluster_distance)
print("Средняя сумма межкластерных расстояний:", average_inter_cluster_distance)

# Подготовка списков для сохранения метрик
avg_cluster_distances = []
avg_intra_cluster_distances = []
avg_inter_cluster_distances = []

# Проведение кластеризации для различных k от 1 до 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)

    # Расчет средней суммы квадратов расстояний до центроида
    avg_cluster_distances.append(kmeans.inertia_ / len(x))

    # Расчет средней суммы средних внутрикластерных расстояний
    closest, _ = pairwise_distances_argmin_min(x, kmeans.cluster_centers_)
    avg_intra_cluster_distances.append(
        np.mean(np.min(pairwise_distances(x, kmeans.cluster_centers_, metric='euclidean')[closest], axis=1)))

    # Расчет средней суммы межкластерных расстояний
    avg_inter_cluster_distances.append(kmeans.score(x) / len(x))

# Построение графиков
plt.figure('Addiction', figsize=(15, 5))

# Зависимость средней суммы квадратов расстояний до центроида
plt.subplot(1, 3, 1)
plt.plot(range(1, 11), avg_cluster_distances, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Average cluster distances')
plt.title('Average cluster distances vs. Number of clusters')

# Зависимость средней суммы средних внутрикластерных расстояний
plt.subplot(1, 3, 2)
plt.plot(range(1, 11), avg_intra_cluster_distances, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Average intra-cluster distances')
plt.title('Average intra-cluster distances vs. Number of clusters')

# Зависимость средней суммы межкластерных расстояний
plt.subplot(1, 3, 3)
plt.plot(range(1, 11), avg_inter_cluster_distances, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Average inter-cluster distances')
plt.title('Average inter-cluster distances vs. Number of clusters')

plt.show()
