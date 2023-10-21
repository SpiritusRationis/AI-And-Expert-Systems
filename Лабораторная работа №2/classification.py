# -*- coding: utf-8 -*-
"""
<b>Бинарная классификация фактографических данных.</b>
Цель работы: получить практические навыки решения задачи бинарной классификации
данных в среде Jupiter Notebook, научиться загружать данные, обучать
классификаторы и проводить классификацию, научиться оценивать точность
полученных моделей.
"""

# импортируем библиотеки
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap

# plot_2d_separator.py by amueller https://github.com/amueller/mglearn/blob/master/mglearn/plot_2d_separator.py

cm_cycle = ListedColormap(['#0000aa', '#ff5050', '#50ff50', '#9040a0', '#fff000'])
cm3 = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])
cm2 = ListedColormap(['#0000aa', '#ff2020'])

def plot_2d_classification(classifier, X, fill=False, ax=None, eps=None, alpha=1, cm=cm3):
    # multiclass
    if eps is None:
        eps = X.std() / 2.

    if ax is None:
        ax = plt.gca()

    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx = np.linspace(x_min, x_max, 1000)
    yy = np.linspace(y_min, y_max, 1000)

    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    decision_values = classifier.predict(X_grid)
    ax.imshow(decision_values.reshape(X1.shape), extent=(x_min, x_max,
                                                         y_min, y_max),
              aspect='auto', origin='lower', alpha=alpha, cmap=cm)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())


def plot_2d_scores(classifier, X, ax=None, eps=None, alpha=1, cm="viridis",
                   function=None):
    # binary with fill
    if eps is None:
        eps = X.std() / 2.

    if ax is None:
        ax = plt.gca()

    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx = np.linspace(x_min, x_max, 100)
    yy = np.linspace(y_min, y_max, 100)

    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    if function is None:
        function = getattr(classifier, "decision_function",
                           getattr(classifier, "predict_proba"))
    else:
        function = getattr(classifier, function)
    decision_values = function(X_grid)
    if decision_values.ndim > 1 and decision_values.shape[1] > 1:
        # predict_proba
        decision_values = decision_values[:, 1]
    grr = ax.imshow(decision_values.reshape(X1.shape),
                    extent=(x_min, x_max, y_min, y_max), aspect='auto',
                    origin='lower', alpha=alpha, cmap=cm)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    return grr


def plot_2d_separator(classifier, X, fill=False, ax=None, eps=None, alpha=1,
                      cm=cm2, linewidth=None, threshold=None,
                      linestyle="solid"):
    # binary?
    if eps is None:
        eps = X.std() / 2.

    if ax is None:
        ax = plt.gca()

    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx = np.linspace(x_min, x_max, 1000)
    yy = np.linspace(y_min, y_max, 1000)

    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    try:
        decision_values = classifier.decision_function(X_grid)
        levels = [0] if threshold is None else [threshold]
        fill_levels = [decision_values.min()] + levels + [
            decision_values.max()]
    except AttributeError:
        # no decision_function
        decision_values = classifier.predict_proba(X_grid)[:, 1]
        levels = [.5] if threshold is None else [threshold]
        fill_levels = [0] + levels + [1]
    if fill:
        ax.contourf(X1, X2, decision_values.reshape(X1.shape),
                    levels=fill_levels, alpha=alpha, cmap=cm)
    else:
        ax.contour(X1, X2, decision_values.reshape(X1.shape), levels=levels,
                   colors="black", alpha=alpha, linewidths=linewidth,
                   linestyles=linestyle, zorder=5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

# генерируем выборку в соответсвии с вариантом

centers = 2
random_state = 28
cluster_std = 4.5

x, y = make_blobs(centers = centers, random_state = random_state, cluster_std = cluster_std, shuffle = 1)

# выводим первые 15 координат и меток

print ("Координаты точек: ")
print (x[:15])
print ("Метки класса: ")
print (y[:15])

# выводим сгенерированные данные по меткам

plt.scatter (x[:, 0], x[:, 1], c = y)
plt.show()

# Метод к-ближайших соседей

def KNeighbors(x, y, test_size = 0.25, n_neighbors = 5):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 1)
  knn = KNeighborsClassifier(n_neighbors = n_neighbors, metric = 'euclidean')
  knn.fit(x_train, y_train)
  prediction = knn.predict(x_test)

  print ('Prediction and test:')
  print ('Prediction: \t', prediction)
  print ('Test: \t\t', y_test)
  print("\n\n")
  print ('Confusion matrix: ')
  print (confusion_matrix(y_test, prediction)[0])
  print (confusion_matrix(y_test, prediction)[1])
  print("\n\n")
  print ('Accuracy score: ', accuracy_score(prediction, y_test))
  print("\n\n")
  print('Classification Report\n', classification_report(y_test, prediction))
  print("\n\n")
  print('ROC AUC')
  print(roc_auc_score(y_test, prediction))
  print("\n\n")

  # обучающая и тестовая выборки
  plt.title('Division into training (Blue) and test (Red) samples')
  plt.scatter (x_train[:, 0], x_train[:, 1], color = 'blue')
  plt.scatter (x_test[:, 0], x_test[:, 1], color = 'red')
  plt.show()

  plt.xlabel("first feature")
  plt.ylabel("second feature")
  plot_2d_separator(knn, x, fill=True)
  plt.scatter(x[:, 0], x[:, 1], c=y, s=70)

# Наивный байесовский метод

def naiveBayes(x, y, test_size = 0.25):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 1)
  gnb = GaussianNB()
  gnb.fit(x_train, y_train)
  prediction = gnb.predict(x_test)

  print ('Prediction and test:')
  print ('Prediction: \t', prediction)
  print ('Test: \t\t', y_test)
  print("\n\n")
  print ('Confusion matrix: ')
  print (confusion_matrix(y_test, prediction)[0])
  print (confusion_matrix(y_test, prediction)[1])
  print("\n\n")
  print ('Accuracy score: ', accuracy_score(prediction, y_test))
  print("\n\n")
  print('Classification Report\n', classification_report(y_test, prediction))
  print("\n\n")
  print('ROC AUC')
  print(roc_auc_score(y_test, prediction))
  print("\n\n")

  # обучающая и тестовая выборки
  plt.title('Division into training (Blue) and test (Red) samples')
  plt.scatter (x_train[:, 0], x_train[:, 1], color = 'blue')
  plt.scatter (x_test[:, 0], x_test[:, 1], color = 'red')
  plt.show()

  plt.xlabel("first feature")
  plt.ylabel("second feature")
  plot_2d_separator(gnb, x, fill=True)
  plt.scatter(x[:, 0], x[:, 1], c=y, s=70)

# Случайный лес

def randomForest(x, y, test_size = 0.25, n_estimators = 100):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 1)
  rfc = RandomForestClassifier(n_estimators = n_estimators)
  rfc.fit(x_train, y_train)
  prediction = rfc.predict(x_test)

  print ('Prediction and test:')
  print ('Prediction: \t', prediction)
  print ('Test: \t\t', y_test)
  print("\n\n")
  print ('Confusion matrix: ')
  print (confusion_matrix(y_test, prediction)[0])
  print (confusion_matrix(y_test, prediction)[1])
  print("\n\n")
  print ('Accuracy score: ', accuracy_score(prediction, y_test))
  print("\n\n")
  print('Classification Report\n', classification_report(y_test, prediction))
  print("\n\n")
  print('ROC AUC')
  print(roc_auc_score(y_test, prediction))
  print("\n\n")

  # обучающая и тестовая выборки
  plt.title('Division into training (Blue) and test (Red) samples')
  plt.scatter (x_train[:, 0], x_train[:, 1], color = 'blue')
  plt.scatter (x_test[:, 0], x_test[:, 1], color = 'red')
  plt.show()

  plt.xlabel("first feature")
  plt.ylabel("second feature")
  plot_2d_separator(rfc, x, fill=True)
  plt.scatter(x[:, 0], x[:, 1], c=y, s=70)

"""Метод к-ближайших соседей (1, 3, 5, 9)"""

KNeighbors(x, y, n_neighbors = 1)

KNeighbors(x, y, n_neighbors = 3)

KNeighbors(x, y, n_neighbors = 5)

KNeighbors(x, y, n_neighbors = 9)

"""Наивный байесовский метод"""

naiveBayes(x, y)

"""Случайный лес (5, 10, 15, 20, 50)"""

randomForest(x, y, n_estimators = 5)

randomForest(x, y, n_estimators = 10)

randomForest(x, y, n_estimators = 15)

randomForest(x, y, n_estimators = 20)

randomForest(x, y, n_estimators = 50)

"""Тестовая часть — 10% выборки"""

KNeighbors(x, y, test_size = 0.1)

naiveBayes(x, y, test_size = 0.1)

randomForest(x, y, test_size = 0.1)

"""Тестовая часть — 35% выборки"""

KNeighbors(x, y, test_size = 0.35)

naiveBayes(x, y, test_size = 0.35)

randomForest(x, y, test_size = 0.35)

randomForest(x, y, test_size = 0.35, n_estimators = 10)