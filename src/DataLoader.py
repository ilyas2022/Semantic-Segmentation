# DataLoader.py
import os
from torch.utils.data import DataLoader
from dataset import CitySegmentationDataset
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_dataset_paths(images_dir, labels_dir):
    """Verifica que los directorios existen y contienen archivos."""
    logger.info(f"Verificando el directorio de imágenes: {images_dir}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"El directorio de imágenes no existe: {images_dir}")
    
    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    logger.info(f"Imágenes encontradas en {images_dir}: {len(images)}")

    logger.info(f"Verificando el directorio de etiquetas: {labels_dir}")
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"El directorio de etiquetas no existe: {labels_dir}")
    
    labels = [f for f in os.listdir(labels_dir) if f.endswith('.png')]
    logger.info(f"Etiquetas correspondientes encontradas en {labels_dir}: {len(labels)}")

    return len(images), len(labels)

def create_dataloader(images_dir, labels_dir, target_size=(256, 256), batch_size=4, shuffle=True, num_workers=0):
    """Crea un DataLoader para el dataset de segmentación de ciudades."""
    num_images, num_labels = verify_dataset_paths(images_dir, labels_dir)
    
    dataset = CitySegmentationDataset(
        images=images_dir,
        labels=labels_dir,
        target_size=target_size
    )
    
    # Crear dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader

if __name__ == "__main__":
    # Crear el DataLoader con rutas relativas
    train_dataloader = create_dataloader(
        images_dir="data/processed/images",
        labels_dir="data/processed/labels",
        batch_size=4
    )
    
    # Probar el DataLoader
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        logger.info(f"Batch {batch_idx + 1}:")
        logger.info(f"Imágenes shape: {images.shape}")
        logger.info(f"Etiquetas shape: {labels.shape}")
        break  # Salir después de un lote