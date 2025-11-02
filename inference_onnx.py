import argparse
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO

def visualize_keypoints(image_path, results, output_path=None):
    """Визуализация ключевых точек на изображении в стиле датасета"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
        
    padding = 40
    img_padded = cv2.copyMakeBorder(img_rgb, padding, padding, padding, padding, 
                                   cv2.BORDER_CONSTANT, value=(50, 50, 50))
        
    colors = {
        'TL': (255, 0, 0),    # Красный
        'TR': (0, 255, 0),    # Зеленый  
        'BR': (0, 0, 255),    # Синий
        'BL': (255, 255, 0)   # Голубой
    }
        
    for r in results:
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            for kpts in r.keypoints.xy:    
                keypoints = kpts.cpu().numpy()
                                
                for i, (label, color) in enumerate(zip(['TL', 'TR', 'BR', 'BL'], colors.values())):
                    if i < len(keypoints):
                        x, y = keypoints[i]
                                        
                        px = int(x) + padding
                        py = int(y) + padding
                        
                        cv2.circle(img_padded, (px, py), 8, color, -1)
                                                
                        if label == 'TL':
                            text_x, text_y = px - 25, py - 10
                        elif label == 'TR':
                            text_x, text_y = px + 10, py - 10
                        elif label == 'BR':
                            text_x, text_y = px + 10, py + 20
                        elif label == 'BL':
                            text_x, text_y = px - 25, py + 20
                                               
                        cv2.putText(img_padded, label, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    plt.figure(figsize=(10, 10))
    plt.imshow(img_padded)
    plt.axis('off')
    plt.title(f"Детекция ключевых точек номерного знака", fontsize=14)
    plt.tight_layout()
        
    if output_path:    
        output_bgr = cv2.cvtColor(img_padded, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), output_bgr)
        print(f"Результат сохранен: {output_path}")
    
    plt.show()

def print_keypoints_info(results):    
    for i, r in enumerate(results):
        print(f"\nРезультат {i+1}:")
        
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            print(f"  Найдено ключевых точек: {len(r.keypoints.xy)}")
            
            for j, kpts in enumerate(r.keypoints.xy):
                print(f"  Объект {j+1}:")
                keypoints = kpts.cpu().numpy()
                
                for k, (label, (x, y)) in enumerate(zip(['TL', 'TR', 'BR', 'BL'], keypoints)):
                    print(f"    {label}: ({x:.1f}, {y:.1f})")
        else:
            print("  Ключевые точки не найдены")

def main():
    parser = argparse.ArgumentParser(description='Инференс YOLO11n-pose модели для детекции ключевых точек номеров')
    parser.add_argument('--model', type=str, required=True, help='Путь к ONNX модели')
    parser.add_argument('--image', type=str, required=True, help='Путь к входному изображению')
    parser.add_argument('--output', type=str, default=None, help='Путь для сохранения результата (опционально)')
    parser.add_argument('--conf', type=float, default=0.5, help='Порог confidence для детекции')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Модель не найдена: {args.model}")
    
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Изображение не найдено: {args.image}")
            
    model = YOLO(args.model, task = 'pose')
    results = model(args.image, conf=args.conf, imgsz=640)
    
    print_keypoints_info(results)    
    visualize_keypoints(args.image, results, args.output)

if __name__ == "__main__":
    main()