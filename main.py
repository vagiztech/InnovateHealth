# Версия python 3.8.10 64-bit
# Импортируем необходимые библиотеки
import numpy as np             # Библиотека для работы с массивами данных
import pandas as pd            # Библиотека для работы с данными в табличной форме
import seaborn as sb           # Библиотека для визуализации данных
import matplotlib.pyplot as plt # Библиотека для создания графиков

# Импортируем дополнительные библиотеки и модули
from glob import glob          # Для работы с путями к файлам
from PIL import Image          # Для работы с изображениями
from sklearn.model_selection import train_test_split  # Для разделения данных на обучающую и тестовую выборки

# Импортируем библиотеки TensorFlow и Keras
import tensorflow as tf
from tensorflow import keras
from keras import layers
from functools import partial

# Настраиваем автоматическую оптимизацию для работы с данными
AUTO = tf.data.experimental.AUTOTUNE

# Отключаем предупреждения для чистоты вывода
import warnings
warnings.filterwarnings('ignore')

# Загружаем список файлов из директории 'train_cancer'
images = glob('train_cancer/*/*.jpg')

# Выводим количество найденных изображений
print(len(images))


