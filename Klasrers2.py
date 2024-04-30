import numpy as np
import matplotlib

matplotlib.use('Agg')  # Установка бэкенда Agg для Matplotlib
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import fcluster

# Создаем набор данных для демонстрации
data, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.60, random_state=0)


# Функция для выполнения иерархической кластеризации и сохранения дендрограммы в файл
def hierarchical_clustering_and_save(data, method='ward', filename='dendrogram.png'):
    # Создание матрицы расстояний
    linkage_matrix = sch.linkage(data, method=method)

    # Построение дендрограммы
    plt.figure(figsize=(10, 7))
    sch.dendrogram(linkage_matrix)
    plt.title(f'Hierarchical Clustering Dendrogram using {method} linkage')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.savefig(filename)  # Сохранение дендрограммы в файл
    plt.close()  # Закрытие фигуры, чтобы освободить память

    # Получение меток кластера на основе заданного расстояния
    max_d = 5  # Максимальное расстояние для разделения на кластеры
    clusters = fcluster(linkage_matrix, max_d, criterion='distance')

    # Визуализация результатов кластеризации на исходных данных
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='prism')  # цвета выбраны по кластерам
    plt.title(f'Cluster plot using {method} linkage')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(f'clusters_{method}.png')
    plt.close()


# Сохранение дендрограмм и визуализация кластеров для различных методов схожести
linkage_methods = ['single', 'complete', 'average', 'ward']
for method in linkage_methods:
    hierarchical_clustering_and_save(data, method=method, filename=f'dendrogram_{method}.png')
