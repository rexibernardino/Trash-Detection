import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. Setup Model (Sama seperti sebelumnya)
def get_model():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    
    # Pastikan path model benar
    model_path = 'TrashDetection\\best_trashnet_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: File model '{model_path}' tidak ditemukan di folder ini!")
        exit()
        
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 2. Transformasi (Sama)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Fungsi Prediksi
def predict(image_path, model):
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # Membersihkan path dari tanda kutip jika user copy-paste path Windows
    image_path = image_path.replace('"', '').replace("'", "").strip()
    
    if not os.path.exists(image_path):
        return None, "File gambar tidak ditemukan!"
    
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    return classes[predicted_idx.item()], confidence.item() * 100

# 4. Input User
if __name__ == "__main__":
    model = get_model()
    print("\n=== AI Trash Classifier ===")
    user_input = input("Masukkan path lengkap gambar (contoh: C:\\Users\\sampah.jpg): ")
    
    label, conf = predict(user_input, model)
    
    if label:
        print("-" * 30)
        print(f"Hasil Prediksi: {label}")
        print(f"Tingkat Kepercayaan: {conf:.2f}%")
        print("-" * 30)
    else:
        print(f"Error: {conf}")