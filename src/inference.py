import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights
import os
import cv2
import time  # Añadido para medir tiempo

def create_class_colormap():
    """Crear mapa de colores exactos para visualización"""
    return {
        0: (107, 142, 35),   # árboles - verde oliva
        1: (128, 64, 128),   # carretera - morado
        2: (70, 70, 70),     # edificios - gris
        3: (70, 130, 180)    # cielo - azul acero
    }

def predict_image(model, image_path, device, target_size=(512, 512)):
    """Realizar predicción en una imagen"""
    # Cargar y preprocesar imagen
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Redimensionar
    image = image.resize(target_size, Image.Resampling.BILINEAR)
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # Predicción
    model.eval()
    with torch.no_grad():
        # Medir tiempo de inferencia
        start_time = time.time()
        output = model(image_tensor.to(device))['out']
        inference_time = time.time() - start_time
        
        # Aplicar softmax para obtener probabilidades
        probabilities = torch.nn.functional.softmax(output, dim=1)
        max_probs, prediction = torch.max(probabilities, dim=1)
        prediction = prediction.squeeze().cpu().numpy()
        max_probs = max_probs.squeeze().cpu().numpy()
        low_confidence = max_probs < 0.8
        prediction[low_confidence] = 255
    
    return image, prediction, original_size, inference_time

def save_predictions(image, prediction, output_dir, filename):
    """Guardar resultados de la predicción"""
    # Crear directorios si no existen
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualization'), exist_ok=True)
    
    # Crear máscara coloreada
    colored_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    class_colormap = create_class_colormap()
    
    # Asignar colores solo a las clases definidas
    for class_idx, color in class_colormap.items():
        mask = prediction == class_idx
        colored_mask[mask] = color
    
    # El resto queda en negro (0, 0, 0)
    
    # Crear superposición
    overlay = np.array(image) * 0.6 + colored_mask * 0.4
    
    # Guardar máscara de predicción
    prediction_img = Image.fromarray(colored_mask)
    prediction_img.save(os.path.join(output_dir, 'labels', f'{filename}_pred.png'))
    
    # Guardar visualización
    plt.figure(figsize=(15, 5))
    
    # Imagen original
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    
    # Predicción coloreada
    plt.subplot(132)
    plt.imshow(colored_mask)
    plt.title('Segmentación')
    plt.axis('off')
    
    # Superposición
    plt.subplot(133)
    plt.imshow(overlay.astype(np.uint8))
    plt.title('Superposición')
    plt.axis('off')
    
    # Guardar visualización
    plt.savefig(os.path.join(output_dir, 'visualization', f'{filename}_vis.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def predict_frame(model, frame, device, target_size=(512, 512)):
    """Realizar predicción en un frame de video"""
    # Convertir BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convertir a PIL Image
    image = Image.fromarray(frame_rgb)
    original_size = image.size
    
    # Usar la misma lógica que predict_image
    image = image.resize(target_size, Image.Resampling.BILINEAR)
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))['out']
        probabilities = torch.nn.functional.softmax(output, dim=1)
        max_probs, prediction = torch.max(probabilities, dim=1)
        prediction = prediction.squeeze().cpu().numpy()
        max_probs = max_probs.squeeze().cpu().numpy()
        low_confidence = max_probs < 0.8
        prediction[low_confidence] = 255
    
    return image, prediction, original_size

def process_video(model, video_path, output_dir, device):
    """Procesar video frame por frame"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return
    
    # Obtener información del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Configurar el writer para el video de salida
    filename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, 'visualization', f'{filename}_processed.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Procesar frame
        image, prediction, _ = predict_frame(model, frame, device)
        
        # Crear máscara coloreada
        colored_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        class_colormap = create_class_colormap()
        for class_idx, color in class_colormap.items():
            mask = prediction == class_idx
            colored_mask[mask] = color
            
        # Redimensionar la máscara al tamaño original del frame
        colored_mask = cv2.resize(colored_mask, (width, height))
        
        # Combinar frame original con máscara
        combined = np.hstack((frame, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)))
        out.write(combined)
        
        frame_count += 1
        if frame_count % 30 == 0:  # Mostrar progreso cada 30 frames
            print(f"Frames procesados: {frame_count}")
    
    cap.release()
    out.release()
    print(f"Video procesado guardado en: {output_path}")

def main():
    # Configuración
    if not torch.cuda.is_available():
        raise RuntimeError("Este script requiere una GPU con CUDA")
    
    device = torch.device("cuda")  # Forzar uso de GPU
    print(f"Usando dispositivo: {device}")
    
    # Directorios
    test_dir = "data/test/images"
    output_dir = "data/test/results"
    
    # Cargar modelo
    model = deeplabv3_resnet50(weights=None)  # Sin pesos preentrenados
    model.classifier[-1] = torch.nn.Conv2d(256, 4, kernel_size=1)
    
    try:
        checkpoint = torch.load('models/best_model.pth', weights_only=True)
        state_dict = checkpoint['model_state_dict']
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aux_classifier')}
        model.load_state_dict(state_dict, strict=False)
        print("Modelo cargado correctamente")
        print(f"Época: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    except Exception as e:
        print(f"Error cargando el modelo: {str(e)}")
        return
    
    model = model.to(device)
    
    # Procesar imágenes y videos
    total_time = 0
    num_images = 0
    
    for file in os.listdir(test_dir):
        if file.endswith(('.jpg', '.png')):
            # Procesar imagen
            image_path = os.path.join(test_dir, file)
            print(f"\nProcesando imagen: {file}")
            image, prediction, original_size, inference_time = predict_image(model, image_path, device)
            filename = os.path.splitext(file)[0]
            save_predictions(image, prediction, output_dir, filename)
            
            # Mostrar tiempo de inferencia
            print(f"Tiempo de inferencia: {inference_time:.3f} segundos")
            total_time += inference_time
            num_images += 1
        
        elif file.endswith(('.mp4', '.avi')):
            # Procesar video
            video_path = os.path.join(test_dir, file)
            print(f"\nProcesando video: {file}")
            process_video(model, video_path, output_dir, device)
    
    # Mostrar estadísticas de tiempo
    if num_images > 0:
        avg_time = total_time / num_images
        print(f"\nEstadísticas de tiempo:")
        print(f"Tiempo total: {total_time:.3f} segundos")
        print(f"Tiempo promedio por imagen: {avg_time:.3f} segundos")
        print(f"FPS promedio: {1/avg_time:.2f}")
    
    print("Procesamiento completado")

if __name__ == "__main__":
    main()