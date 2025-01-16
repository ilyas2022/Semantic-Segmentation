import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights
import logging
import os
from dataset import CitySegmentationDataset

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_class_mapping():
    """Mapeo de clases a índices"""
    return {
        'arboles': 0,
        'carretera': 1,
        'edificios': 2,
        'cielo': 3
    }

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=True):
            outputs = model(images)['out']
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if i % 10 == 0:
            logger.info(f'Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}')
            
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")
    
    # Parámetros
    num_classes = 4  # arboles, carretera, edificios, cielo
    batch_size = 8
    epochs = 50
    learning_rate = 1e-4
    
    # Directorios de datos
    data_dir = "data/processed"
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    # Crear dataset
    dataset = CitySegmentationDataset(
        images=images_dir,
        labels=labels_dir,
        target_size=(512, 512),
        num_classes=num_classes
    )
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Modelo con pesos actualizados
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    model = model.to(device)
    
    # Optimizador y criterio
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # Directorio para guardar modelos
    os.makedirs('models', exist_ok=True)
    
    # Entrenamiento
    best_val_loss = float('inf')
    for epoch in range(epochs):
        logger.info(f"\nÉpoca {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        logger.info(f"Loss de entrenamiento: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        logger.info(f"Loss de validación: {val_loss:.4f}")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'models/best_model.pth')
            logger.info("Guardado nuevo mejor modelo")

if __name__ == "__main__":
    main()
