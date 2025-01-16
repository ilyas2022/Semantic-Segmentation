import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class EvaluadorSegmentacion:
    def __init__(self):
        self.clases = {
            0: "árboles",
            1: "carretera",
            2: "edificios",
            3: "cielo"
        }
        self.resultados = {}
        
    def calcular_exactitud(self, pred_mask, true_mask):
        """Calcula la exactitud por clase y global"""
        exactitud_por_clase = {}
        total_pixels = pred_mask.size
        
        for clase_id, nombre_clase in self.clases.items():
            pred_binaria = (pred_mask == clase_id)
            true_binaria = (true_mask == clase_id)
            
            interseccion = np.logical_and(pred_binaria, true_binaria).sum()
            union = np.logical_or(pred_binaria, true_binaria).sum()
            
            iou = interseccion / union if union > 0 else 0
            exactitud = (pred_binaria == true_binaria).sum() / total_pixels
            
            exactitud_por_clase[nombre_clase] = {
                'exactitud': exactitud * 100,
                'iou': iou * 100
            }
        
        exactitud_global = (pred_mask == true_mask).sum() / total_pixels * 100
        return exactitud_por_clase, exactitud_global
    
    def evaluar_prediccion(self, pred_path, true_path):
        """Evalúa una predicción contra su ground truth"""
        try:
            pred_mask = np.array(Image.open(pred_path))
            true_mask = np.array(Image.open(true_path))
            
            exactitud_por_clase, exactitud_global = self.calcular_exactitud(pred_mask, true_mask)
            
            self.resultados = {
                'por_clase': exactitud_por_clase,
                'global': exactitud_global
            }
            
            return self.resultados
        except Exception as e:
            print(f"Error procesando {os.path.basename(pred_path)}: {str(e)}")
            return None
    
    def mostrar_resultados(self, nombre_imagen=""):
        """Muestra los resultados de forma visual"""
        if not self.resultados:
            print("No hay resultados para mostrar")
            return
        
        # Gráfico de barras
        clases = list(self.resultados['por_clase'].keys())
        exactitudes = [self.resultados['por_clase'][clase]['exactitud'] for clase in clases]
        ious = [self.resultados['por_clase'][clase]['iou'] for clase in clases]
        
        x = np.arange(len(clases))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, exactitudes, width, label='Exactitud')
        rects2 = ax.bar(x + width/2, ious, width, label='IoU')
        
        ax.set_ylabel('Porcentaje')
        ax.set_title(f'Métricas por Clase - {nombre_imagen}')
        ax.set_xticks(x)
        ax.set_xticklabels(clases, rotation=45)
        ax.legend()
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.savefig(f'resultados_{nombre_imagen}.png')
        plt.close()
        
        # Resultados en consola
        print(f"\nResultados para {nombre_imagen}:")
        print(f"Exactitud Global: {self.resultados['global']:.2f}%")
        print("\nPor clase:")
        for clase, metricas in self.resultados['por_clase'].items():
            print(f"{clase}:")
            print(f"  Exactitud: {metricas['exactitud']:.2f}%")
            print(f"  IoU: {metricas['iou']:.2f}%")

def procesar_todas_las_imagenes():
    evaluador = EvaluadorSegmentacion()
    
    # Directorios
    test_dir = "data/test/images"
    pred_dir = "data/test/results/labels"
    true_dir = "data/processed/labels"
    
    # Crear directorio de resultados si no existe
    os.makedirs("resultados_evaluacion", exist_ok=True)
    
    # Resultados globales
    resultados_globales = []
    
    # Procesar cada imagen
    for imagen in os.listdir(test_dir):
        if imagen.endswith(('.jpg', '.png')):
            nombre_base = os.path.splitext(imagen)[0]
            print(f"\nProcesando: {nombre_base}")
            
            # Construir rutas
            pred_path = os.path.join(pred_dir, f"{nombre_base}_pred.png")
            true_path = os.path.join(true_dir, f"{nombre_base}.png")
            
            # Verificar archivos
            if not os.path.exists(pred_path):
                print(f"No se encuentra la predicción: {pred_path}")
                continue
            if not os.path.exists(true_path):
                print(f"No se encuentra el ground truth: {true_path}")
                continue
            
            # Evaluar imagen
            resultados = evaluador.evaluar_prediccion(pred_path, true_path)
            if resultados:
                evaluador.mostrar_resultados(nombre_base)
                resultados_globales.append({
                    'imagen': nombre_base,
                    'resultados': resultados
                })
    
    # Mostrar resumen global
    if resultados_globales:
        print("\n=== RESUMEN GLOBAL ===")
        exactitud_global_total = 0
        for resultado in resultados_globales:
            exactitud_global_total += resultado['resultados']['global']
        
        exactitud_promedio = exactitud_global_total / len(resultados_globales)
        print(f"\nExactitud global promedio: {exactitud_promedio:.2f}%")
        
        # Guardar resultados en un archivo
        with open("resultados_evaluacion/resumen.txt", "w") as f:
            f.write("=== RESUMEN DE EVALUACIÓN ===\n\n")
            f.write(f"Exactitud global promedio: {exactitud_promedio:.2f}%\n\n")
            for resultado in resultados_globales:
                f.write(f"\nImagen: {resultado['imagen']}\n")
                f.write(f"Exactitud global: {resultado['resultados']['global']:.2f}%\n")
                f.write("Por clase:\n")
                for clase, metricas in resultado['resultados']['por_clase'].items():
                    f.write(f"  {clase}:\n")
                    f.write(f"    Exactitud: {metricas['exactitud']:.2f}%\n")
                    f.write(f"    IoU: {metricas['iou']:.2f}%\n")

if __name__ == "__main__":
    procesar_todas_las_imagenes() 