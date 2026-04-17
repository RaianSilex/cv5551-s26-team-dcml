
import argparse
from pathlib import Path
from ultralytics import YOLO

BASE_MODEL = 'yolov8n.pt'


def train(data_yaml: str, epochs: int, imgsz: int, batch: int, project: str):
    print(f'Loading base model: {BASE_MODEL}')
    model = YOLO(BASE_MODEL)

    print(f'Starting training on: {data_yaml}')
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name='gesture_yolo',
        exist_ok=True,

        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        flipud=0.0, 
        fliplr=0.3,  
        mosaic=0.5,
        mixup=0.1,


        patience=10,
    )

    best_weights = Path(project) / 'gesture_yolo' / 'weights' / 'best.pt'
    print(f'\nTraining complete.')
    print(f'Best weights: {best_weights}')
    print(f'\nNext step:')
    print(f'  Update YOLO_GESTURE_MODEL in gesture_input.py to: {best_weights}')
    return best_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    required=True,              help='Path to dataset.yaml')
    parser.add_argument('--epochs',  type=int, default=30)
    parser.add_argument('--imgsz',   type=int, default=640,      help='Input image size')
    parser.add_argument('--batch',   type=int, default=16)
    parser.add_argument('--project', default='runs',             help='Output folder')
    args = parser.parse_args()

    train(args.data, args.epochs, args.imgsz, args.batch, args.project)
