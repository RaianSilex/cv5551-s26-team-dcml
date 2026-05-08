import cv2, json, time
from xarm.wrapper import XArmAPI

from utils.zed_camera import ZedCamera
from utils.vis_utils import draw_pose_axes
from checkpoint1 import GRIPPER_LENGTH, CUBE_TAG_SIZE
from config import ROBOT_IP, INGREDIENT_TAG_MAP
from primitives import ContainerDetector, execute_add_ingredient, execute_stir
from task_planner import get_task_plan_from_detected


def execute_plan(arm, plan, poses, log=print):
    for i, step in enumerate(plan):
        action = step['action']
        label  = f'[Step {i+1}/{len(plan)}] {action}'
        if 'ingredient' in step:
            label += f' — {step["ingredient"]}'
        log(label)

        if action == 'ADD_INGREDIENT':
            ingredient = step['ingredient'].lower()
            if ingredient not in INGREDIENT_TAG_MAP:
                log(f'  [SKIP] Unknown ingredient: {ingredient}')
                continue
            success = execute_add_ingredient(arm, ingredient, poses)
            if not success:
                log(f'  [ABORT] Failed to add {ingredient}.')
                return False

        elif action == 'STIR':
            success = execute_stir(arm, poses)
            if not success:
                log(f'  [ABORT] Failed to stir.')
                return False

        else:
            log(f'  [SKIP] Unknown action: {action}')

    log('Beverage preparation complete!')
    return True


def run_beverage_task(user_requirement='', confirm=True, log=print):

    zed              = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic
    detector         = ContainerDetector(camera_intrinsic)

    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        log('Capturing scene...')
        cv_image    = zed.image
        depth_image = zed.depth
        point_cloud = zed.point_cloud   # HxWx4 XYZW in meters

        log('Detecting containers...')
        poses, poses_cam = detector.detect_all(cv_image, depth_image, point_cloud)
        if poses is None or len(poses) == 0:
            return {'status': 'error', 'message': 'No containers detected.', 'response': None}

        log(f'Detected IDs: {list(poses.keys())}')

        detected_ingredients = [
            name for name, tag_id in INGREDIENT_TAG_MAP.items()
            if tag_id in poses
        ]
        log(f'Detected ingredients: {detected_ingredients}')

        for t_cam_obj in poses_cam.values():
            draw_pose_axes(cv_image, camera_intrinsic, t_cam_obj, size=CUBE_TAG_SIZE)

        log(f'User request: {user_requirement!r}')
        log('Sending to GPT-4o for task planning...')
        response = get_task_plan_from_detected(detected_ingredients, user_requirement)
        log('Received response:')
        log(json.dumps(response, indent=2))

        if response.get('status') != 'ok':
            message = response.get('message', 'Unknown planning error.')
            return {'status': 'error', 'message': message, 'response': response}

        plan     = response.get('plan', [])
        beverage = response.get('beverage', '?')

        if confirm:
            log(f'About to make: {beverage}. Press "k" to execute, any other key to abort.')
            cv2.namedWindow('Beverage Setup', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Beverage Setup', 1280, 720)
            cv2.imshow('Beverage Setup', cv_image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key != ord('k'):
                return {'status': 'error', 'message': 'Aborted by user.', 'response': response}

        log(f'Executing plan for {beverage}...')
        success = execute_plan(arm, plan, poses, log=log)
        if not success:
            return {'status': 'error', 'message': 'Execution failed mid-plan.', 'response': response}

        return {'status': 'ok', 'message': f'{beverage} prepared successfully.', 'response': response}

    finally:
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()
        zed.close()


def main():
    user_requirement = input('What would you like? (e.g. "coffee, lactose intolerant"): ')
    result = run_beverage_task(user_requirement, confirm=True)
    print(f'\nFinal status: {result["status"]}')
    print(f'Message: {result["message"]}')


if __name__ == '__main__':
    main()