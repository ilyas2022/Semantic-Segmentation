import os
import numpy as np
from PIL import Image
from collections import Counter

def analyze_labels(labels_dir):
    print("=== Análisis de Etiquetas ===\n")
    
    # Obtener lista de archivos de etiquetas
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])
    
    # Diccionario para almacenar colores únicos
    color_counts = Counter()
    
    # Analizar cada etiqueta
    print("Analizando etiquetas...")
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        label = Image.open(label_path)
        label_array = np.array(label)
        
        # Obtener colores únicos
        unique_values = np.unique(label_array.reshape(-1, label_array.shape[-1] if len(label_array.shape) > 2 else 1), axis=0)
        for value in unique_values:
            if len(value.shape) > 0:
                color_counts[tuple(value)] += 1
            else:
                color_counts[int(value)] += 1
    
    print("\nColores únicos encontrados:")
    for color, count in color_counts.most_common():
        if isinstance(color, tuple):
            print(f"Color RGB{color}: aparece en {count} imágenes")
        else:
            print(f"Valor {color}: aparece en {count} imágenes")
    
    return color_counts

def main():
    labels_dir = os.path.join("data", "processed", "labels")
    color_counts = analyze_labels(labels_dir)

if __name__ == "__main__":
    main()