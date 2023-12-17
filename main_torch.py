# Импортируем необходимые библиотеки
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
import os
import shutil
###############################
# Загрузка и подготовка данных#
###############################
# !kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
# !unzip -q skin-cancer-mnist-ham10000.zip -d data_cancer/
# Пути к файлам и папкам
# csv_file = 'data_cancer/HAM10000_metadata.csv'
# images_dir_1 = 'data_cancer/HAM10000_images_part_1'
# images_dir_2 = 'data_cancer/HAM10000_images_part_2'

# # Чтение CSV-файла
# metadata = pd.read_csv(csv_file)

# # Перебор строк в DataFrame 1
# for index, row in metadata.iterrows():
#     image_name = row['image_id'] + '.jpg'  # предполагаем, что расширение файлов .jpg
#     class_name = row['dx']
#     source_path = os.path.join(images_dir_1, image_name)
#     target_dir = os.path.join(images_dir_1, class_name)

#     # Проверка наличия файла и перемещение
#     if os.path.exists(source_path):
#         os.makedirs(target_dir, exist_ok=True)
#         shutil.move(source_path, os.path.join(target_dir, image_name))

# # Перебор строк в DataFrame 2
# for index, row in metadata.iterrows():
#     image_name = row['image_id'] + '.jpg'  # предполагаем, что расширение файлов .jpg
#     class_name = row['dx']
#     source_path = os.path.join(images_dir_2, image_name)
#     target_dir = os.path.join(images_dir_2, class_name)

#     # Проверка наличия файла и перемещение
#     if os.path.exists(source_path):
#         os.makedirs(target_dir, exist_ok=True)
#         shutil.move(source_path, os.path.join(target_dir, image_name))


# Определяем преобразования для предобработки изображений
transform = transforms.Compose([
    transforms.Resize(256),        # Изменяем размер изображения до 256x256 пикселей
    transforms.CenterCrop(224),    # Обрезаем изображение до размера 224x224 пикселей
    transforms.ToTensor(),         # Преобразуем изображение в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Нормализуем тензор
])

# Создаем DataLoader для обучающего набора данных
train_dataset = ImageFolder(root='data_cancer/HAM10000_images_part_1', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Создаем DataLoader для валидационного набора данных
valid_dataset = ImageFolder(root='data_cancer/HAM10000_images_part_2', transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

# Определяем модель GoogLeNet с предварительно обученными весами
def google():
    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(1024, len(train_dataset.classes))  # Заменяем последний слой
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

# Определяем функцию обучения модели
def train(model, optimizer, train_loader, val_loader, epoch=10):
    loss_train, acc_train = [], []
    loss_valid, acc_valid = [], []
    for epoch in tqdm(range(epoch)):
        losses, equals = [], []
        torch.set_grad_enabled(True)

        # Обучаем модель
        model.train()
        for i, (image, target) in enumerate(train_loader):
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss = criterion(output, target)

            losses.append(loss.item())
            equals.extend([x.item() for x in torch.argmax(output, 1) == target])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_train.append(np.mean(losses))
        acc_train.append(np.mean(equals))
        losses, equals = [], []
        torch.set_grad_enabled(False)

        # Валидируем модель
        model.eval()
        for i, (image, target) in enumerate(valid_loader):
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss = criterion(output, target)

            losses.append(loss.item())
            equals.extend([y.item() for y in torch.argmax(output, 1) == target])

        loss_valid.append(np.mean(losses))
        acc_valid.append(np.mean(equals))

    return loss_train, acc_train, loss_valid, acc_valid

# Проверяем доступность GPU
use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'

# Определяем функцию потерь (кросс-энтропия)
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)

# Создаем модель
model = google()
print('Model: GoogLeNet\n')

# Определяем оптимизатор
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model = model.to(device)

# Обучаем модель и сохраняем статистику
loss_train, acc_train, loss_valid, acc_valid = train(model, optimizer, train_loader, valid_loader, 30)

# Выводим статистику точности
print('acc_train:', acc_train, '\nacc_valid:', acc_valid)

# Очищаем память GPU и удаляем модель
del model
torch.cuda.empty_cache()
