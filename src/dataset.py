# dataset.py
import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset

class CitySegmentationDataset(Dataset):
    def __init__(self, images, labels, target_size=(512, 512), num_classes=4):
        self.images_dir = images
        self.labels_dir = labels
        self.target_size = target_size
        self.num_classes = num_classes
        
        # Mapeo de colores RGB a índices de clase
        self.color_to_class = {
            (107, 142, 35): 0,   # árboles
            (128, 64, 128): 1,   # carretera
            (70, 70, 70): 2,     # edificios
            (70, 130, 180): 3    # cielo
        }
        
        # Obtener lista de archivos
        self.image_files = sorted([f for f in os.listdir(images) if f.endswith('.jpg')])
        self.label_files = sorted([f for f in os.listdir(labels) if f.endswith('.png')])
        
        print(f"Imágenes encontradas: {len(self.image_files)}")
        print(f"Etiquetas encontradas: {len(self.label_files)}")
    
    def __getitem__(self, idx):
        # Cargar imagen y etiqueta
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        # Cargar imagen
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.target_size, Image.Resampling.BILINEAR)
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        
        # Cargar etiqueta
        label = Image.open(label_path).convert('RGB')
        label = label.resize(self.target_size, Image.Resampling.NEAREST)
        label = np.array(label)
        
        # Crear máscara de clases
        label_mask = np.zeros(self.target_size, dtype=np.int64)
        
        # Convertir colores RGB a índices de clase
        for color, class_idx in self.color_to_class.items():
            mask = np.all(np.abs(label - np.array(color)) < 30, axis=2)
            label_mask[mask] = class_idx
        
        # Convertir a tensor
        label_mask = torch.from_numpy(label_mask)
        
        return image, label_mask
    
    def __len__(self):
        return len(self.image_files)
