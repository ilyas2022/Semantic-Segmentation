import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def create_label_mapping():
    """Define el mapeo de colores exactos a clases"""
    return {
        'arboles': (107, 142, 35),    # Verde oliva
        'carretera': (128, 64, 128),  # Morado azulado
        'edificios': (70, 70, 70),    # Gris
        'cielo': (70, 130, 180)       # Azul acero
    }

def preprocess_dataset(input_images_dir, input_labels_dir, output_images_dir, output_labels_dir, target_size=(512, 512)):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    class_colors = create_label_mapping()
    print("Clases a detectar:")
    for clase, color in class_colors.items():
        print(f"- {clase}: RGB{color}")
    
    image_files = sorted([f for f in os.listdir(input_images_dir) if f.endswith('.jpg')])
    label_files = sorted([f for f in os.listdir(input_labels_dir) if f.endswith('.png')])
    
    print(f"\nProcesando {len(image_files)} imágenes...")
    
    for img_file, label_file in tqdm(zip(image_files, label_files)):
        try:
            # Procesar imagen
            img_path = os.path.join(input_images_dir, img_file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Procesar etiqueta
            label_path = os.path.join(input_labels_dir, label_file)
            label = Image.open(label_path).convert('RGB')
            label = label.resize(target_size, Image.Resampling.NEAREST)
            label = np.array(label)
            
            # Crear etiqueta procesada
            label_processed = np.zeros((*target_size, 3), dtype=np.uint8)
            
            # Tolerancia para la detección de colores
            tolerance = 20
            
            # Procesar cada clase
            for clase, color in class_colors.items():
                color_array = np.array(color)
                # Crear máscara para cada color
                mask = np.all(np.abs(label - color_array) < tolerance, axis=2)
                label_processed[mask] = color
            
            # Guardar resultados
            output_img_path = os.path.join(output_images_dir, img_file)
            output_label_path = os.path.join(output_labels_dir, label_file)
            
            Image.fromarray(image_resized).save(output_img_path)
            Image.fromarray(label_processed).save(output_label_path)
            
            # Debug: imprimir colores únicos
            unique_colors = np.unique(label_processed.reshape(-1, 3), axis=0)
            if len(unique_colors) < 2:
                print(f"\nAdvertencia: {label_file} tiene pocos colores: {unique_colors}")
                
        except Exception as e:
            print(f"\nError procesando {img_file} y {label_file}: {str(e)}")
            continue

def main():
    input_images_dir = os.path.join("data", "raw", "images")
    input_labels_dir = os.path.join("data", "raw", "labels")
    output_images_dir = os.path.join("data", "processed", "images")
    output_labels_dir = os.path.join("data", "processed", "labels")
    
    preprocess_dataset(input_images_dir, input_labels_dir, output_images_dir, output_labels_dir)
    print("\n¡Preprocesamiento completado!")

if __name__ == "__main__":
    main()
