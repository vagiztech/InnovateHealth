import os
import torch
from torchvision import models, transforms
from PIL import Image

# Загрузка модели и весов
def load_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = google()
    model = model.to(device)
    
    # Загрузка весов
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    return model
# Модель GoogLeNet
num_classes = 7
def google(): # pretrained=True для tensorflow
    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(1024, 7)
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
# Функция для классификации изображения
def classify_image(image_path, model):
    input_tensor = load_and_preprocess_image(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    # Передача изображения через модель
    with torch.no_grad():
        output = model(input_tensor)
    
    # Получение предсказанного класса
    _, predicted_class = torch.max(output, 1)
    
    return predicted_class.item()

# Функция для загрузки и предобработки изображения
def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path)
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    
    return input_batch

# Загрузка обученной модели
weights_path = '/content/weights30.pt'
model = torch.load(weights_path)

# Путь к папке с изображениями для классификации
folder_path = '/content/data_cancer/HAM10000_images_part_1'

predicted_class_name = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# Классификация изображений в папке
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)
    if os.path.isfile(image_path):
        predicted_class = classify_image(image_path, model)
        print(f"Изображение {filename} классифицировано как рак типа: {predicted_class_name[predicted_class]}")