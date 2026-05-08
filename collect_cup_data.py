import cv2
import os
import time
from utils.zed_camera import ZedCamera

SAVE_DIR = './data/images'
os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    zed = ZedCamera()

    existing = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')])
    count    = existing
    print(f'Starting from image {count} (found {existing} existing)')
    print()
    print('Controls:')
    print('  SPACE  — save current frame')
    print('  n      — skip (next frame without saving)')
    print('  q      — quit')
    print()
    print('Tips:')
    print('  - Rearrange cups between captures for variety')
    print('  - Aim for 50-100 images total')
    print('  - Include shots with some cups close together')
    print('  - Include shots with only 3-4 cups visible')

    cv2.namedWindow('Collect Data', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Collect Data', 1280, 720)

    try:
        while True:
            frame = zed.image

            if len(frame.shape) > 2 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            display = frame.copy()
            cv2.putText(display, f'Saved: {count} images',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display, 'SPACE=save  n=skip  q=quit',
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow('Collect Data', display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                path = os.path.join(SAVE_DIR, f'cup_{count:04d}.jpg')
                cv2.imwrite(path, frame)
                count += 1
                print(f'Saved {path}')
                border = frame.copy()
                cv2.rectangle(border, (0,0), (border.shape[1]-1, border.shape[0]-1),
                              (0,255,0), 20)
                cv2.imshow('Collect Data', border)
                cv2.waitKey(200)

    finally:
        cv2.destroyAllWindows()
        zed.close()
        print(f'\nDone. Collected {count - existing} new images ({count} total).')
        print(f'Images saved to: {SAVE_DIR}')
        print()

if __name__ == '__main__':
    main()
