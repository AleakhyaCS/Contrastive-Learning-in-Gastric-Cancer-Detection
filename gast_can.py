import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import pickle

class ContrastiveModel(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.encoder = models.resnet50(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)

class DownstreamClassifier(nn.Module):
    def __init__(self, encoder, embedding_dim=128, hidden_dim=128, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        for name, module in self.model.named_modules():
            # safer match: endswith target_layer
            if name.endswith(self.target_layer):
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        # compute Grad-CAM
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.size(3), input_tensor.size(2)))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def preprocess_image(img, size=224):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor

def overlay_cam_on_image(img, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.applyColorMap(np.uint8(255*cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(img), 1-alpha, heatmap, alpha, 0)
    return overlay

# Streamlit

st.title("Gastric Cancer Detection: TUM vs NOR")
st.write("Upload an image to predict Tumor (TUM) or Normal (NOR) tissue and visualize Grad-CAM.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_tensor = preprocess_image(img).to(device)
    input_tensor.requires_grad_(True)

    # Load models
    encoder = ContrastiveModel().to(device)
    encoder.load_state_dict(torch.load("cl_encoder_resnet50_wo_str_trail3.pth", map_location=device))
    encoder.eval()

    classifier = DownstreamClassifier(encoder, num_classes=2).to(device)
    classifier.load_state_dict(torch.load("finetuned model.pth", map_location=device))
    classifier.eval()

    # Grad-CAM
    grad_cam = GradCAM(classifier.encoder, target_layer="layer4")
    cam = grad_cam.generate(input_tensor)
    overlay_img = overlay_cam_on_image(img, cam)
    grad_cam.remove_hooks()

    # Prediction
    with torch.no_grad():
        outputs = classifier(input_tensor)
        pred_class = outputs.argmax(dim=1).item()
        class_names = ["NOR", "TUM"]
        st.write(f"Prediction: {class_names[pred_class]}")

    if pred_class == 1:  # If TUM
        with open("clus_pipeline.pkl", "rb") as f:
            cluster_data = pickle.load(f)

        scaler = cluster_data["scaler"]
        rf = cluster_data["rf_classifier"]

        # Extract embedding using the CL encoder
        with torch.no_grad():
            embedding = encoder(input_tensor).cpu().numpy()   # shape (1, 128)

        # Scale embedding
        embedding_scaled = scaler.transform(embedding)

        # Predict Cluster Category
        cluster_rf = int(rf.predict(embedding_scaled)[0])
        st.write(f"Cluster Category: {cluster_rf}")

    st.image(overlay_img, caption="Grad-CAM Heatmap Overlay", use_column_width=True)
