import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Load Model (dengan Cache agar tidak loading terus-menerus)
@st.cache_resource
def load_model():
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    model.load_state_dict(torch.load('TrashDetection\\best_trashnet_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# 2. Setup Transformasi (Harus SAMA dengan training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. UI Streamlit
st.set_page_config(page_title="Trash Detection AI", page_icon="🗑️")
st.title("🗑️ Trash Recognition AI")
st.write("Upload foto sampah, dan model AI kita akan mengidentifikasi jenisnya!")

model = load_model()
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar terpilih', use_container_width=True)
    
    # Proses Prediksi
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    # Hasil
    result = classes[predicted_idx.item()]
    conf_score = confidence.item() * 100
    
    st.markdown(f"### Prediksi: **{result}**")
    st.progress(conf_score/100)
    st.write(f"Tingkat Kepercayaan: **{conf_score:.2f}%**")