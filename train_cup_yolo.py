
import argparse
from pathlib import Path
from ultralytics import YOLO


def train(data_yaml, epochs, imgsz, batch, device, project):
    print(f'Loading base model: yolov8n.pt')
    model = YOLO('yolov8n.pt')

    print(f'Training on: {data_yaml}')
    print(f'Device: {device}')

    results = model.train(
        data    = data_yaml,
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        device  = device,
        project = project,
        name    = 'cup_yolo',
        exist_ok= True,
        patience= 15,

        flipud  = 0.0,    
        fliplr  = 0.5,    
        hsv_h   = 0.02,   
        hsv_s   = 0.5,   
        hsv_v   = 0.4,    
        mosaic  = 0.5,
        degrees = 10.0,   
    )

    best = Path(project) / 'cup_yolo' / 'weights' / 'best.pt'
    print(f'\nTraining complete.')
    print(f'Best weights: {best}')
    print(f'\nNext step: update CUP_MODEL_PATH in primitives.py to: {best}')
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    required=True,            help='Path to data.yaml from Roboflow')
    parser.add_argument('--epochs',  type=int,   default=50)
    parser.add_argument('--imgsz',   type=int,   default=640)
    parser.add_argument('--batch',   type=int,   default=16)
    parser.add_argument('--device',  default='0',              help='cuda device (0) or cpu')
    parser.add_argument('--project', default='runs',           help='Output folder')
    args = parser.parse_args()

    train(args.data, args.epochs, args.imgsz, args.batch, args.device, args.project)
