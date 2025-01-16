import torch
import sys

def check_cuda():
    print("\n=== Verificación de CUDA ===")
    
    # Versiones
    print(f"\nVersión de Python: {sys.version}")
    print(f"Versión de PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Versión de CUDA: {torch.version.cuda}")
        print(f"\nGPU detectada: {torch.cuda.get_device_name(0)}")
        
        # Prueba simple
        print("\nRealizando prueba de GPU...")
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = torch.matmul(x, y)
        
        print("✓ Operación en GPU exitosa")
        print(f"Memoria GPU usada: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    else:
        print("\n❌ CUDA no está disponible")
        print("Verifica la instalación de PyTorch y los drivers de NVIDIA")

if __name__ == "__main__":
    check_cuda()