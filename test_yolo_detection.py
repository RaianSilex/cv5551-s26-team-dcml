import argparse, sys, cv2
from ultralytics import YOLO, YOLOWorld


CUP_CLASSES = [
    'ivory cup',
    'light blue cup',
    'yellow cup',
    'orange cup',
    'green cup'
]
CONF_THRESHOLD = 0.3


def stage1(image_path='IMG_5030.jpg'):
    print('\n' + '='*50)
    print('STAGE 1 — YOLO-World on saved photo')
    print('='*50)

    img = cv2.imread(image_path)
    if img is None:
        print(f'[ERROR] Could not load image: {image_path}')
        return False

    model = '/home/rob/FPv3/cv5551-s26-team-dcml/best.pt'
    model.set_classes(CUP_CLASSES)

    results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]

    detected = []
    if results.boxes is not None:
        for box in results.boxes:
            label = model.names[int(box.cls)]
            conf  = float(box.conf)
            detected.append((label, conf))
            print(f'  {label}: {conf:.2f}')

    if not detected:
        print('  [WARN] No cups detected. Try lowering conf or tweaking color prompts.')
    else:
        print(f'  Detected {len(detected)}/{len(CUP_CLASSES)} cup classes.')

    out_path = 'stage1_result.jpg'
    cv2.imwrite(out_path, results.plot())
    print(f'  Annotated image saved to {out_path}')

    missing = set(CUP_CLASSES) - {d[0] for d in detected}
    if missing:
        print(f'  [WARN] Missing: {missing}')
        print('  Try tweaking the color prompt for those cups.')

    return len(detected) > 0


def stage2():
    print('\n' + '='*50)
    print('STAGE 2 — YOLO-World on live ZED frame')
    print('='*50)

    try:
        from utils.zed_camera import ZedCamera
    except ImportError:
        print('[ERROR] ZED camera utils not found. Run this at the university machine.')
        return False

    zed = ZedCamera()

    try:
        model = YOLO('/home/rob/FPv3/cv5551-s26-team-dcml/best.pt')
        #model = YOLOWorld('yolov8s-worldv2.pt')
        #model.set_classes(CUP_CLASSES)

        print('  Capturing ZED frame...')
        frame   = zed.image
        
        if len(frame.shape) > 2 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        detected = []
        if results.boxes is not None:
            for box in results.boxes:
                label = model.names[int(box.cls)]
                conf  = float(box.conf)
                detected.append((label, conf))
                print(f'  {label}: {conf:.2f}')

        if not detected:
            print('  [WARN] No cups detected in ZED frame.')
        else:
            print(f'  Detected {len(detected)}/{len(CUP_CLASSES)} cup classes.')

        out_path = 'stage2_result.jpg'
        cv2.imwrite(out_path, results.plot())
        print(f'  Annotated image saved to {out_path}')

        return len(detected) > 0

    finally:
        zed.close()


def stage3():
    print('\n' + '='*50)
    print('STAGE 3 — Full ContainerDetector (depth + robot transform)')
    print('='*50)

    try:
        from utils.zed_camera import ZedCamera
        from primitives import ContainerDetector
        from config import INGREDIENT_TAG_MAP, STIRRER_TAG_ID, STIRRER_POSE_KEY
    except ImportError as e:
        print(f'[ERROR] Import failed: {e}')
        return False

    zed = ZedCamera()
    try:
        camera_intrinsic = zed.camera_intrinsic
        detector         = ContainerDetector(camera_intrinsic)

        print('  Capturing ZED frame + depth...')
        frame       = zed.image
        depth       = zed.depth
        point_cloud = zed.point_cloud
        poses, poses_cam = detector.detect_all(frame, depth, point_cloud)
        if poses is None:
            print('  [ERROR] Camera-to-robot transform failed. Check table AprilTags.')
            return False

        if len(poses) == 0:
            print('  [WARN] No objects detected.')
            return False

        print(f'\n  Detected {len(poses)} objects:')
        for tag_id, pose in poses.items():
            x = pose[0,3] * 1000
            y = pose[1,3] * 1000
            z = pose[2,3] * 1000

            name = next((n for n, t in INGREDIENT_TAG_MAP.items() if t == tag_id), None)
            if name is None and tag_id == STIRRER_POSE_KEY:
                name = 'stirrer'
            name = name or f'id={tag_id}'

            plausible = 50 < abs(x) < 1000 and abs(y) < 1000 and 0 < z < 500
            status    = 'OK' if plausible else 'CHECK — values look off'
            print(f'  {name:15s} x={x:7.1f} y={y:7.1f} z={z:7.1f} mm  [{status}]')

        print('\n  Expected ingredients:')
        for name, tag_id in INGREDIENT_TAG_MAP.items():
            found = 'FOUND' if tag_id in poses else 'MISSING'
            print(f'    {name:15s} (id={tag_id}): {found}')
        stirrer_status = 'FOUND' if STIRRER_POSE_KEY in poses else 'MISSING'
        print(f'    {"stirrer":15s} (id={STIRRER_TAG_ID}): {stirrer_status}')

        return True

    finally:
        zed.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, choices=[1, 2, 3],
                        help='Which stage to run (default: all)')
    parser.add_argument('--image', default='IMG_5030.jpg',
                        help='Image path for stage 1 (default: IMG_5030.jpg)')
    args = parser.parse_args()

    stages = [args.stage] if args.stage else [1, 2, 3]

    results = {}
    for s in stages:
        if s == 1:
            results[1] = stage1(args.image)
        elif s == 2:
            results[2] = stage2()
        elif s == 3:
            results[3] = stage3()

    print('\n' + '='*50)
    print('SUMMARY')
    print('='*50)
    labels = {1: 'Photo detection', 2: 'ZED live detection', 3: 'Full pipeline'}
    for s, passed in results.items():
        status = 'PASS' if passed else 'FAIL'
        print(f'  Stage {s} ({labels[s]}): {status}')

    if all(results.values()):
        print('\nAll stages passed. Ready to run test_primitives.py.')
    else:
        print('\nFix failing stages before running the robot.')


if __name__ == '__main__':
    main()
