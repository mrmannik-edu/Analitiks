import plotly.graph_objs as go
import pandas as pd
import time
from collections import defaultdict

# Определение функции для выполнения алгоритма Apriori
def apriori(data, min_support, order):
    # Преобразование набора данных в список транзакций
    transactions = data.values.tolist()

    # Инициализация переменных
    itemset = set()
    freq_sets = defaultdict(int)

    # Создание начального набора элементов
    for transaction in transactions:
        for item in transaction:
            itemset.add(frozenset([item]))
            freq_sets[frozenset([item])] += 1

    # Фильтрация набора элементов на основе min_support
    itemset = set([item for item in itemset if freq_sets[item] >= min_support])

    # Инициализация списка для хранения частых наборов элементов
    freq_itemsets = []

    # Основной цикл для генерации частых наборов элементов
    k = 2
    while itemset:
        # Генерация новых комбинаций элементов
        new_combinations = set([i.union(j) for i in itemset for j in itemset if len(i.union(j)) == k])
        # Подсчет частоты новых комбинаций
        for transaction in transactions:
            for combination in new_combinations:
                if combination.issubset(transaction):
                    freq_sets[combination] += 1
        # Фильтрация на основе min_support
        itemset = set([item for item in new_combinations if freq_sets[item] >= min_support])
        # Добавление в список частых наборов элементов
        freq_itemsets.extend([(item, freq_sets[item]) for item in itemset])
        k += 1

    # Сортировка частых наборов элементов на основе параметра order
    if order == 'support':
        freq_itemsets.sort(key=lambda x: x[1], reverse=True)
    elif order == 'lexicographic':
        freq_itemsets.sort(key=lambda x: tuple(x[0]))

    return freq_itemsets

# Загрузка набора данных
data = pd.read_csv(r'C:\Users\ПК\Desktop\baskets.csv', encoding='ANSI')

# Список для хранения результатов
support_thresholds = [0.01, 0.03, 0.05, 0.10, 0.15]
performance_times = []
frequent_itemsets_counts = []

# Выполнение экспериментов
for support in support_thresholds:
    start_time = time.time()
    frequent_itemsets = apriori(data, support * len(data), 'support')  # Убедитесь, что функция apriori возвращает значение
    end_time = time.time()

    performance_times.append(end_time - start_time)
    if frequent_itemsets is not None:  # Проверка, что frequent_itemsets не является None
        frequent_itemsets_counts.append(len(frequent_itemsets))  # Теперь переменная frequent_itemsets определена
    else:
        frequent_itemsets_counts.append(0)  # Если frequent_itemsets является None, добавляем 0

# Визуализация быстродействия алгоритма с помощью Plotly
performance_trace = go.Scatter(x=support_thresholds, y=performance_times, mode='lines+markers', name='Время выполнения')
layout = go.Layout(title='Быстродействие алгоритма Apriori', xaxis=dict(title='Порог поддержки'), yaxis=dict(title='Время выполнения (секунды)'))
fig = go.Figure(data=[performance_trace], layout=layout)
fig.show()

# Визуализация количества частых наборов различной длины с помощью Plotly
itemsets_trace = go.Bar(x=support_thresholds, y=frequent_itemsets_counts, name='Количество частых наборов')
layout = go.Layout(title='Количество частых наборов при разных порогах поддержки', xaxis=dict(title='Порог поддержки'), yaxis=dict(title='Количество частых наборов'))
fig = go.Figure(data=[itemsets_trace], layout=layout)
fig.show()
