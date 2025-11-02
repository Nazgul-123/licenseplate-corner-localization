# Локализация угловых точек автомобильных номеров

## Описание задачи

Обучить нейронную сеть для локализации угловых точек автомобильных номеров на изображениях. Модель должна определять координаты четырех угловых точек номерного знака (TL, TR, BR, BL).

## Данные

- **Формат изображений**: одноканальные (grayscale), разные разрешения
- **Типы номеров**: однострочные (RECT) и двустрочные (SQUARE)
- **Разметка**: текстовые файлы с относительными координатами точек
- **Формат разметки**:
  ```
  TL X1 Y1
  TR X2 Y2  
  BR X3 Y3
  BL X4 Y4
  ```

## Архитектура проекта

```
├── models/                    # Обученные модели
│   ├── keypoint_model_dynamic.onnx
│   └── yolo11n_pose.onnx
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_yolo_training.ipynb  
│   └── 03_mobilenet_training.ipynb
├── inference_onnx.py         # Скрипт для инференса
└── utils/
    └── evaluation.py         # Функции оценки моделей
```

## Модели

### 1. YOLO11n-pose
- **Архитектура**: YOLO с Pose Estimation
- **Входное разрешение**: 640x640
- **Точность**: MAE 2.03 px, RMSE 2.84 px
- **Скорость**: 26.16 ms (PyTorch)

### 2. Custom MobileNetV3
- **Архитектура**: MobileNetV3-small + регрессия ключевых точек
- **Входное разрешение**: 224x224
- **Точность**: MAE 5.26 px, RMSE 6.68 px
- **Скорость**: 14.58 ms (PyTorch)

## Результаты

### Сравнение производительности

**YOLO11n:**
```
PyTorch:  MAE 2.03 px, Speed 26.16 ms
ONNX:     MAE 2.03 px, Speed 103.47 ms
```

**MobileNetV3:**
```
PyTorch:  MAE 5.26 px, Speed 14.58 ms  
ONNX:     MAE 5.26 px, Speed 21.85 ms
```

## Использование

### Инференс на одном изображении

```bash
python inference_onnx.py --model models/yolo11n_pose.onnx --image test_image.png --output result.png
```

## Выводы

1. **YOLO11n** показала лучшую точность (MAE 2.03 px), но работает медленнее и имеет больший размер модели
2. **MobileNetV3** быстрее (14.58 ms), но менее точна (MAE 5.26 px)  

Лучшая модель: **YOLO11n-pose** (сохранена в ONNX формате в `models/yolo11n_pose.onnx`) 
PS. в скрипте инференса использована данная модель