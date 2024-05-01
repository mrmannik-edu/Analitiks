import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


# Функция для загрузки данных
def load_data(file_path, skiprows=None):
    data = pd.read_csv(file_path, header=None, delimiter=',', skiprows=skiprows)
    # Удаление пробелов в начале и конце строк в каждом столбце
    data = data.apply(lambda col: col.str.strip() if col.dtypes == object else col)
    return data

# Путь к файлам
path_train = r'C:\Users\ПК\Downloads\adult.data.txt'
path_test = r'C:\Users\ПК\Downloads\adult.test.txt'

try:
    # Загрузка тренировочных и тестовых данных
    train_data = load_data(path_train)
    test_data = load_data(path_test, skiprows=1)

    # Назначение имен столбцов
    columns = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num', 'Marital-Status',
               'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-Gain', 'Capital-Loss',
               'Hours-per-week', 'Native-Country', 'Income']
    train_data.columns = columns
    test_data.columns = columns

    # Инициализация LabelEncoder
    le = LabelEncoder()

    # Предобработка категориальных данных с LabelEncoder
    cat_features = ['Workclass', 'Education', 'Marital-Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Native-Country']
    for col in cat_features:
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = le.transform(test_data[col])

    # Кодирование целевой переменной, убедитесь что 'Income' чист от пробелов и точек
    train_data['Income'] = le.fit_transform(train_data['Income'].str.replace('.', ''))
    test_data['Income'] = le.transform(test_data['Income'].str.replace('.', ''))

    # Подготовка данных для обучения
    X_train = train_data.drop('Income', axis=1)
    y_train = train_data['Income']
    X_test = test_data.drop('Income', axis=1)
    y_test = test_data['Income']

    # Создание модели дерева решений
    model = DecisionTreeClassifier(criterion='gini')
    model.fit(X_train, y_train)

    # Предсказание на тестовом наборе
    predictions = model.predict(X_test)

    # Оценка модели
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='binary', pos_label=1)
    recall = recall_score(y_test, predictions, average='binary', pos_label=1)
    f1 = f1_score(y_test, predictions, average='binary', pos_label=1)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

except Exception as e:
    print("Произошла ошибка при загрузке или обработке данных:", e)


import matplotlib
matplotlib.use('Agg')  # Использование бэкенда Agg, который подходит для сохранения изображений в файл
import matplotlib.pyplot as plt


# Значения метрик
metrics = {
    'Accuracy': 0.8104,
    'Precision': 0.5970,
    'Recall': 0.6071,
    'F1 Score': 0.6020
}

# Названия метрик и их значения
names = list(metrics.keys())
values = list(metrics.values())

# Создание столбчатой диаграммы
plt.figure(figsize=(10, 5))
plt.bar(names, values, color=['blue', 'green', 'red', 'purple'])

# Добавление заголовка и меток
plt.title('Performance Metrics of the Decision Tree Model')
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.ylim([0, 1])  # Установка пределов оси Y для лучшего сравнения

# Сохранение графика в файл
plt.savefig('metrics_plot.png')
