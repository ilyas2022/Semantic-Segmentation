import os
from PIL import Image
import numpy as np

def verify_processed_labels():
    labels_dir = "data/processed/labels"
    print("Verificando etiquetas procesadas...")
    
    valid_colors = {
        (107, 142, 35),   # árboles
        (128, 64, 128),   # carretera
        (70, 70, 70),     # edificios
        (70, 130, 180),   # cielo
        (0, 0, 0)         # fondo/negro
    }
    
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.png'):
            label_path = os.path.join(labels_dir, label_file)
            label = np.array(Image.open(label_path))
            
            # Obtener colores únicos
            unique_colors = set(map(tuple, label.reshape(-1, 3)))
            
            # Verificar colores no válidos
            invalid_colors = unique_colors - valid_colors
            if invalid_colors:
                print(f"Colores no válidos en {label_file}: {invalid_colors}")

if __name__ == "__main__":
    verify_processed_labels()