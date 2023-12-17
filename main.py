# Версия python 3.8.10 64-bit
# tensorflow==2.7.0
# Импортируем необходимые библиотеки
import numpy as np             # Библиотека для работы с массивами данных
import pandas as pd            # Библиотека для работы с данными в табличной форме
import seaborn as sb           # Библиотека для визуализации данных
import matplotlib.pyplot as plt # Библиотека для создания графиков
import os

# Импортируем дополнительные библиотеки и модули
from glob import glob          # Для работы с путями к файлам
from PIL import Image          # Для работы с изображениями
from sklearn.model_selection import train_test_split  # Для разделения данных на обучающую и тестовую выборки

# Импортируем библиотеки TensorFlow и Keras
import tensorflow as tf
from tensorflow import keras
from keras import layers
from functools import partial

# Получаем текущую директорию скрипта и устанавливаем её как рабочую
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

# Настраиваем автоматическую оптимизацию для работы с данными
AUTO = tf.data.experimental.AUTOTUNE

# Отключаем предупреждения для чистоты вывода
import warnings
warnings.filterwarnings('ignore')

# Загружаем список файлов из директории 'train_cancer'
images = glob('train_cancer/*/*.jpg')

# Выводим количество найденных изображений
#print(len(images))
# Замена обратных слешей на прямые в путях к изображениям
images = [path.replace('\\', '/') for path in images]
df = pd.DataFrame({'filepath': images}) # Создание DataFrame
df['label'] = df['filepath'].str.split('/', expand=True)[1]
#print(df.head())

df['label_bin'] = np.where(df['label'].values == 'malignant', 1, 0) # Создание бинарных меток (0 - benign, 1 - malignant)
# df.head()

# Визуализация распределения меток в виде круговой диаграммы
# x = df['label'].value_counts()

# plt.pie(x.values,
# 		labels=x.index,
# 		autopct='%1.1f%%')
#plt.show()


# Отображение случайных изображений для каждой категории
# for cat in df['label'].unique():
# 	temp = df[df['label'] == cat]

# 	index_list = temp.index
# 	fig, ax = plt.subplots(1, 4, figsize=(15, 5))
# 	fig.suptitle(f'Images for {cat} category . . . .', fontsize=20)
# 	for i in range(4):
# 		index = np.random.randint(0, len(index_list))
# 		index = index_list[index]
# 		data = df.iloc[index]

# 		image_path = data[0]

# 		img = np.array(Image.open(image_path))
# 		ax[i].imshow(img)
# plt.tight_layout()
#plt.show()


# Подготовка данных для обучения модели
features = df['filepath']
target = df['label_bin']

X_train, X_val,\
	Y_train, Y_val = train_test_split(features, target,
									test_size=0.15,
									random_state=10)

X_train.shape, X_val.shape


# Создание функции для декодирования изображений и меток
def decode_image(filepath, label=None):
    Metka = 1
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    # if label == 'benign':
    #     Metka = 0
    # else:
    #     Metka = 1
    # return img, Metka


# Создание TensorFlow Dataset для обучающей и валидационной выборок
# train_ds = (
# 	tf.data.Dataset
# 	.from_tensor_slices((X_train, Y_train))
# 	.map(decode_image, num_parallel_calls=AUTO)
# 	.batch(32)
# 	.prefetch(AUTO)
# )

#val_ds = (
#	tf.data.Dataset
#	.from_tensor_slices((X_val, Y_val))
#	.map(decode_image, num_parallel_calls=AUTO)
#	.batch(32)
#	.prefetch(AUTO)
#)
#####################################
##############Model##################
#####################################

from tensorflow.keras.applications.efficientnet import EfficientNetB7

# pre_trained_model = EfficientNetB7(
# 	input_shape=(224, 224, 3),
# 	weights='imagenet',
# 	include_top=False
# )

# for layer in pre_trained_model.layers:
# 	layer.trainable = False



from tensorflow.keras import Model

# inputs = layers.Input(shape=(224, 224, 3))
# x = layers.Flatten()(inputs)

# x = layers.Dense(256, activation='relu')(x)
# x = layers.BatchNormalization()(x)
# x = layers.Dense(256, activation='relu')(x)
# x = layers.Dropout(0.3)(x)
# x = layers.BatchNormalization()(x)
# outputs = layers.Dense(1, activation='sigmoid')(x)

# model = Model(inputs, outputs)

