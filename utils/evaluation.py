"""
Функции для оценки моделей детекции ключевых точек.
"""

from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from ultralytics import YOLO
import onnxruntime as ort


def calculate_keypoint_mae(pred_kpts, gt_kpts):
    """Вычисляет MAE между предсказанными и ground truth точками."""
    errors = np.linalg.norm(pred_kpts - gt_kpts, axis=1)
    return errors


def evaluate_yolo_model(model_path, test_images_dir, test_labels_dir):
    """
    Оценка YOLO модели на тестовой выборке.
    
    Returns:
        dict: метрики (mae, rmse, detection_rate, failed_detections)
    """
    model = YOLO(model_path, task = 'pose')
    
    mae_list = []
    rmse_list = []
    detected = 0
    total = 0
    failed_detections = []
    
    for img_file in Path(test_images_dir).glob('*.png'):
        label_file = Path(test_labels_dir) / f"{img_file.stem}.txt"
        if not label_file.exists():
            continue
        
        total += 1
        results = model(str(img_file), verbose=False)
        
        if results[0].keypoints is None or len(results[0].keypoints) == 0:
            failed_detections.append(str(img_file))
            continue
        
        detected += 1
        pred_kpts = results[0].keypoints.xy[0].cpu().numpy()
        
        with open(label_file, 'r') as f:
            line = f.readline().strip().split()
        gt_kpts = np.array([float(x) for x in line[5:]]).reshape(-1, 3)[:, :2]
        
        img_h, img_w = results[0].orig_shape
        gt_kpts[:, 0] *= img_w
        gt_kpts[:, 1] *= img_h
        
        errors = calculate_keypoint_mae(pred_kpts, gt_kpts)
        mae_list.extend(errors)
        rmse_list.extend(errors ** 2)
    
    mae = np.mean(mae_list) if mae_list else float('inf')
    rmse = np.sqrt(np.mean(rmse_list)) if rmse_list else float('inf')
    detection_rate = detected / total if total > 0 else 0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'detection_rate': detection_rate,
        'detected': detected,
        'total': total,
        'failed_detections': failed_detections
    }


def evaluate_pytorch_model(model, test_loader, device):
    """
    Оценка PyTorch модели на тестовой выборке.
    
    Returns:
        dict: метрики (mae, rmse)
    """
    model.eval()
    mae_list = []
    rmse_list = []
    
    with torch.no_grad():
        for imgs, kpts in test_loader:
            imgs, kpts = imgs.to(device), kpts.to(device)
            pred = model(imgs)
            
            pred = pred.cpu().numpy().reshape(-1, 4, 2)
            kpts = kpts.cpu().numpy().reshape(-1, 4, 2)
            
            errors = np.linalg.norm(pred - kpts, axis=2)
            mae_list.extend(errors.flatten())
            rmse_list.extend(errors.flatten() ** 2)
    
    mae = np.mean(mae_list)
    rmse = np.sqrt(np.mean(rmse_list))
    
    return {'mae': mae, 'rmse': rmse}


def evaluate_onnx_model(onnx_path, test_loader):
    """
    Оценка ONNX модели на тестовой выборке.
    
    Returns:
        dict: метрики (mae, rmse)
    """
    session = ort.InferenceSession(
        onnx_path, 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    mae_list = []
    rmse_list = []
    
    for imgs, kpts in test_loader:
        imgs_np = imgs.numpy()
        kpts_np = kpts.numpy()
        
        pred = session.run(None, {'input': imgs_np})[0]
        pred = pred.reshape(-1, 4, 2)
        kpts_np = kpts_np.reshape(-1, 4, 2)
        
        errors = np.linalg.norm(pred - kpts_np, axis=2)
        mae_list.extend(errors.flatten())
        rmse_list.extend(errors.flatten() ** 2)
    
    mae = np.mean(mae_list)
    rmse = np.sqrt(np.mean(rmse_list))
    
    return {'mae': mae, 'rmse': rmse}


def measure_yolo_speed(model_path, test_images_dir, n_images=20):
    """
    Измеряет скорость инференса YOLO модели.
    
    Returns:
        dict: среднее время и std в миллисекундах
    """
    model = YOLO(model_path, task = 'pose')
    test_images = list(Path(test_images_dir).glob('*.png'))[:n_images]
    
    times = []
    for img in test_images:
        start = time.time()
        _ = model(str(img), verbose=False)
        times.append(time.time() - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000
    }


def measure_pytorch_speed(model, test_loader, device, n_batches=10):
    """
    Измеряет скорость инференса PyTorch модели.
    
    Returns:
        dict: среднее время на батч в миллисекундах
    """
    model.eval()
    times = []
    
    with torch.no_grad():
        for idx, (imgs, _) in enumerate(test_loader):
            if idx >= n_batches:
                break
            
            imgs = imgs.to(device)
            start = time.time()
            _ = model(imgs)
            times.append(time.time() - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000
    }


def measure_onnx_speed(onnx_path, test_loader, n_batches=10):
    """
    Измеряет скорость инференса ONNX модели.
    
    Returns:
        dict: среднее время на батч в миллисекундах
    """
    session = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    times = []
    for idx, (imgs, _) in enumerate(test_loader):
        if idx >= n_batches:
            break
        
        imgs_np = imgs.numpy()
        start = time.time()
        _ = session.run(None, {'input': imgs_np})
        times.append(time.time() - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000
    }


def visualize_failed_detections(failed_paths, max_display=5):
    """Визуализирует изображения с неудачными детекциями."""
    if not failed_paths:
        print("✓ Все детекции успешны!")
        return
    
    print(f"Найдено {len(failed_paths)} неудачных детекций")
    
    n_display = min(len(failed_paths), max_display)
    fig, axes = plt.subplots(1, n_display, figsize=(5*n_display, 5))
    
    if n_display == 1:
        axes = [axes]
    
    for idx, path in enumerate(failed_paths[:n_display]):
        img = Image.open(path)
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(Path(path).name, fontsize=8)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def print_metrics(metrics, model_name="Model"):
    """Выводит метрики в красивом формате."""
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    
    if 'mae' in metrics:
        print(f"MAE:            {metrics['mae']:.2f} px")
    if 'rmse' in metrics:
        print(f"RMSE:           {metrics['rmse']:.2f} px")
    if 'detection_rate' in metrics:
        print(f"Detection Rate: {metrics['detection_rate']:.2%} "
              f"({metrics['detected']}/{metrics['total']})")
    if 'mean_ms' in metrics:
        print(f"Inference Time: {metrics['mean_ms']:.2f} ± {metrics['std_ms']:.2f} ms")
    
    print(f"{'='*60}\n")


def compare_models(metrics_dict):
    """
    Выводит сравнительную таблицу метрик.
    
    Args:
        metrics_dict: словарь {model_name: metrics}
    """
    print("\n" + "="*80)
    print(f"{'Model':<25} {'MAE (px)':<12} {'RMSE (px)':<12} {'Speed (ms)':<15}")
    print("-"*80)
    
    for name, metrics in metrics_dict.items():
        mae = f"{metrics['mae']:.2f}" if 'mae' in metrics else "N/A"
        rmse = f"{metrics['rmse']:.2f}" if 'rmse' in metrics else "N/A"
        speed = f"{metrics['mean_ms']:.2f}" if 'mean_ms' in metrics else "N/A"
        
        print(f"{name:<25} {mae:<12} {rmse:<12} {speed:<15}")
    
    print("="*80 + "\n")