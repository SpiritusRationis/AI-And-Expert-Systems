"""
Задать значения количества продаж по 10 товарам в течение 12 месяцев
(помесячно). Для каждого из товаров спрогнозировать количество продаж на следующий, 13-й месяц и провести анализ достоверности планирования продаж.
"""

# Импорт модулей
import numpy as np
import pandas as pd
import seaborn
import random


# Функция для формирования массива продаж для продукта (rnd)
def randomSales(count, min, max):
    li = []
    for i in range(0, count):
        tmp = random.randint(min, max)
        li.append(tmp)
    return li


# задаем количество продаж на 1 год  по продуктам
sales = pd.DataFrame({
    'Laptops': randomSales(12, 10, 35),
    'Smartphones': randomSales(12, 10, 50),
    'Smart Watch': randomSales(12, 10, 40),
    'Computers': randomSales(12, 10, 25),
    'Microwave Ovens': randomSales(12, 10, 30),
    'Refrigerators': randomSales(12, 10, 20),
    'Washing machines': randomSales(12, 10, 15),
    'Dishwashers': randomSales(12, 5, 10),
    'TVs': randomSales(12, 10, 45),
    'Coffee Makers': randomSales(12, 7, 15)
})

print(sales)

# График
lp = seaborn.lineplot(sales)
seaborn.move_legend(lp, 'upper left', bbox_to_anchor=(1, 1))


# p0

p0 = sales.sum() / sales.shape[0]
print(p0)

# Оценка

square_std = ((sales - p0) ** 2).sum() / (sales.shape[0] - 1)
std = square_std ** 0.5
reliability = std / p0
print(reliability)

# Расчёт планируемого показателя

predict = pd.DataFrame([p0 + np.random.normal(0, std, len(p0))])
print(predict)

# Условия

cond_1 = ((sales - p0) < 2 * std).all()
cond_2 = p0 > 2 * std
cond_3 = (sales > 0).all()

print('Условие 1: ')
print(cond_1)
print('\n')
print('Условие 2: ')
print(cond_2)
print('\n')
print('Условие 3: ')
print(cond_3)

# Раскраска товаров
product_color = pd.Series(dtype='string')
for product in sales.columns:
    if not cond_3[product]:
        product_color[product] = 'Red'
    elif not cond_1[product] and not cond_2[product]:
        product_color[product] = 'Orange'
    elif not cond_1[product] or not cond_2[product]:
        product_color[product] = 'Yellow'
    else:
        product_color[product] = 'Green'

print(product_color)
