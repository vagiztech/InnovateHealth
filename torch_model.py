import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, random_split

# Загрузка kaggle API ключа для скачивания данных
from google.colab import files
files.upload()

# Создание папки для kaggle API ключа
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Установка библиотеки kaggle
!pip install kaggle

# Скачивание и распаковка датасета
!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
!unzip -q skin-cancer-mnist-ham10000.zip -d data_cancer/

import os
import shutil

# Пути к файлам и папкам
csv_file = 'data_cancer/HAM10000_metadata.csv'
images_dir_1 = 'data_cancer/HAM10000_images_part_1'
images_dir_2 = 'data_cancer/HAM10000_images_part_2'

# Чтение CSV-файла
metadata = pd.read_csv(csv_file)

# Функция для сортировки данных по папкам
def sort_data_by_folder(csv_file, images_dir, metadata):
    for index, row in metadata.iterrows():
        image_name = row['image_id'] + '.jpg'
        class_name = row['dx']
        source_path = os.path.join(images_dir, image_name)
        target_dir = os.path.join(images_dir, class_name)

        # Проверка наличия файла и перемещение
        if os.path.exists(source_path):
            os.makedirs(target_dir, exist_ok=True)  # Создание папки класса, если не существует
            shutil.move(source_path, os.path.join(target_dir, image_name))

# Сортировка данных по папкам
sort_data_by_folder(csv_file, images_dir_1, metadata)
sort_data_by_folder(csv_file, images_dir_2, metadata)

# Трансформация данных для нейронной сети
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Создание датасета
full_dataset = ImageFolder(root='data_cancer/HAM10000_images_part_2', transform=transform)
num_classes = len(full_dataset.classes)

# Разделение датасета на тренировочную и валидационную выборку
train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

# Загрузчика данных
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

# Модель GoogleNet
def google():
    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(1024, num_classes)
    for param in model.parameters():
        param.requires_grad = True
    model.inception3a.requires_grad = False
    model.inception3b.requires_grad = False
    model.inception4a.requires_grad = False
    model.inception4b.requires_grad = False
    model.inception4c.requires_grad = False
    model.inception4d.requires_grad = False
    model.inception4e.requires_grad = False
    return model

# Функция для обучения модели
def train(model, optimizer, train_loader, val_loader, epoch=10):
    loss_train, acc_train = [], []
    loss_valid, acc_valid = [], []
    for epoch in tqdm(range(epoch)):
        losses, equals = [], []
        torch.set_grad_enabled(True)

        # Обучение модели
        model.train()
        for i, (image, target) in enumerate(train_loader):
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss = criterion(output,target)

            losses.append(loss.item())
            equals.extend(
                [x.item() for x in torch.argmax(output, 1) == target])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_train.append(np.mean(losses))
        acc_train.append(np.mean(equals))
        losses, equals = [], []
        torch.set_grad_enabled(False)

        # Валидация модели
        model.eval()
        for i , (image, target) in enumerate(valid_loader):
            image = image.to(device)
            target = target.to(device)

            output = model(image)
            loss = criterion(output,target)

            losses.append(loss.item())
            equals.extend(
                [y.item() for y in torch.argmax(output, 1) == target])

        loss_valid.append(np.mean(losses))
        acc_valid.append(np.mean(equals))

    return loss_train, acc_train, loss_valid, acc_valid

# Определение устройства (GPU или CPU) для обучения
use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)

# Создание модели
model = google()
print('Model: GoogLeNet\n')

# Оптимизатор для обучения модели
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model = model.to(device)

# Обучение модели
loss_train, acc_train, loss_valid, acc_valid = train(
model, optimizer, train_loader, valid_loader, 30)
print('acc_train:', acc_train, '\nacc_valid:', acc_valid)

import matplotlib.pyplot as plt

# Функция для построения графика потерь
def plot_loss(Loss_train, Loss_val):
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(Loss_train)), Loss_train, color='orange', label='train', linestyle='--')
    plt.plot(range(len(Loss_val)), Loss_val, color='blue', marker='o', label='val')
    plt.legend()
    plt.show()

# Построение графика потерь на обучающем и валидационном наборах данных
plot_loss(loss_train, loss_valid)

# Сохранение модели
torch.save(model, '/weights30.pt')

# Загрузка и предобработка изображения
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # добавляем размерность пакета

    return input_batch

# Функция для предсказывания класса изображения
def predict_image_class(image_path, model):
    input_tensor = load_and_preprocess_image(image_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model = model.to(device)

    model.eval()

    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

# Список классов
predicted_class_name = full_dataset.classes[:]
print(predicted_class_name)

weights_path = '/content/weights30.pt'
model = torch.load(weights_path)

# Путь к изображению для предикта
image_path = "/content/data_cancer/HAM10000_images_part_1/nv/ISIC_0024321.jpg"
def predict_class_image(image_path, model=model):
  predicted_class = predict_image_class(image_path, model)
  return predicted_class_name[predicted_class]
print(f"Модель предсказывает класс: {predict_class_image(image_path)}")


# Функция для тестирования модели на папке с изображениями
def test_model_lul(Patch):
    test_dict = {}
    folder_path = Patch

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            predicted_class = predict_image_class(file_path, model)
            pred = predicted_class_name[predicted_class]
            if pred in test_dict.keys():
                test_dict[pred] += 1
            else:
                test_dict[pred] = 0
    return test_dict

# Пример вызова функции для тестирования модели на папке
test_results = test_model_lul('/content/data_cancer/HAM10000_images_part_1/nv')
print(test_results)

columns = ['pacientID','full name', 'Typeofdisease', 'Cancer']
df_doctor = pd.DataFrame(columns=columns)

# Функция для заполнения следующей строки
def fill_string(values, dataframe):
    new_string = pd.Series(values, index=dataframe.columns)
    dataframe = pd.concat([dataframe, new_string.to_frame().T], ignore_index=True)
    return dataframe

info_dict = {'akies': ['актинический кератоз/болезнь Боуэна L57.0/D04.9', 'предрак', 'дерматолог'],
             'bcc': ['базалиома D48', 'рак', 'онколог'],
             'bkl': ['доброкачественные поражения, подобные кератозу ', 'ок', 'дома'],
             'df': ['дерматофиброма D23.9', 'ок', 'дерматолог'],
             'mel': ['меланома D48', 'рак', 'онколог'],
             'nv': ['меланоцитарный невус', 'ок', 'дома'],
             'vasc': ['сосудистые поражения', 'ок', 'дома']}


def pacient(path):
  global df_doctor
  pacientID = input('введите ваш id: ')
  full_name = input('введите ваше ФИО: ')
  disease = predict_class_image(path)
  Typeofdisease = info_dict[disease][0]
  Cancer_num = info_dict[disease][1]
  doctor = info_dict[disease][2]
  if Cancer_num == 'рак':
    Canser = 2
  elif Cancer_num == 'предрак':
    Canser = 1
  else:
    Canser = 0
  df_doctor = fill_string([pacientID, full_name, Typeofdisease, Canser], df_doctor)
  if doctor == 'дерматолог':
    return 'Запишитесь к врачу дерматологу'
  if doctor == 'онколог':
    return 'Запишитесь к врачу онкологу'
  if doctor == 'дома':
    return 'У вас все впорядке !'

path_pacient = '/content/data_cancer/HAM10000_images_part_2/mel/ISIC_0029913.jpg'
pacient(path_pacient)


print(df_doctor)