import plotly.graph_objs as go
import pandas as pd
import time
from collections import defaultdict
from itertools import combinations

# Определение функции для выполнения алгоритма Apriori
def apriori(data, min_support, order):
    transactions = data.values.tolist()
    itemset = set()
    freq_sets = defaultdict(int)

    # Создание начального набора элементов
    for transaction in transactions:
        for item in transaction:
            itemset.add(frozenset([item]))
            freq_sets[frozenset([item])] += 1

    # Фильтрация набора элементов на основе min_support
    itemset = set([item for item in itemset if freq_sets[item] >= min_support])
    freq_itemsets = []
    k = 2
    while itemset:
        new_combinations = set([i.union(j) for i in itemset for j in itemset if len(i.union(j)) == k])
        for transaction in transactions:
            for combination in new_combinations:
                if combination.issubset(transaction):
                    freq_sets[combination] += 1
        itemset = set([item for item in new_combinations if freq_sets[item] >= min_support])
        freq_itemsets.extend([(item, freq_sets[item]) for item in itemset])
        k += 1

    if order == 'support':
        freq_itemsets.sort(key=lambda x: x[1], reverse=True)
    elif order == 'lexicographic':
        freq_itemsets.sort(key=lambda x: tuple(x[0]))

    return freq_itemsets, freq_sets

# Функция для генерации ассоциативных правил
def generate_rules(freq_itemsets, freq_sets, min_confidence, total_transactions):
    rules = []
    for itemset, support in freq_itemsets:
        for k in range(1, len(itemset)):
            for antecedent in combinations(itemset, k):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                antecedent_support = freq_sets[antecedent] / total_transactions
                rule_confidence = support / total_transactions / antecedent_support
                if rule_confidence >= min_confidence:
                    rules.append((antecedent, consequent, support / total_transactions, rule_confidence))
    return rules

# Загрузка набора данных
data = pd.read_csv(r'C:\Users\ПК\Desktop\baskets.csv', encoding='ANSI')
total_transactions = len(data)

# Выполнение алгоритма Apriori и генерация правил
frequent_itemsets, freq_sets = apriori(data, 0.1 * len(data), 'support')
confidence_levels = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
performance_times = []
rules_count = []

# Проходим по различным уровням достоверности и собираем статистику
for confidence in confidence_levels:
    start_time = time.time()
    rules = generate_rules(frequent_itemsets, freq_sets, confidence, total_transactions)
    end_time = time.time()
    performance_times.append(end_time - start_time)
    rules_count.append(len(rules))

# Печать правил
for antecedent, consequent, support, confidence in rules:
    print(f"{set(antecedent)} → {set(consequent)}, support: {support:.4f}, confidence: {confidence:.4f}")

# Визуализация результатов

# График времени выполнения
time_trace = go.Scatter(x=confidence_levels, y=performance_times, mode='lines+markers', name='Время выполнения')
time_layout = go.Layout(title='Время выполнения по разным уровням достоверности',
                        xaxis={'title': 'Порог достоверности'},
                        yaxis={'title': 'Время, сек'})
time_fig = go.Figure(data=[time_trace], layout=time_layout)
time_fig.show()

# График количества правил
count_trace = go.Bar(x=confidence_levels, y=rules_count, name='Количество правил')
count_layout = go.Layout(title='Количество правил по разным уровням достоверности',
                         xaxis={'title': 'Порог достоверности'},
                         yaxis={'title': 'Количество правил'})
count_fig = go.Figure(data=[count_trace], layout=count_layout)
count_fig.show()
