"""
FP1 – Beverage-Making Robot

Main orchestrator. Exposes `run_beverage_task(user_requirement)` so the GUI
(or any other caller) can trigger the full capture -> plan -> execute flow.
Running this file directly prompts for a request on the command line.
"""

import cv2, json, time
from xarm.wrapper import XArmAPI

from utils.zed_camera import ZedCamera
from utils.vis_utils import draw_pose_axes
from checkpoint1 import GRIPPER_LENGTH, CUBE_TAG_SIZE
from config import ROBOT_IP, INGREDIENT_TAG_MAP
from primitives import ContainerDetector, execute_add_ingredient, execute_stir
from task_planner import get_task_plan


def execute_plan(arm, plan, poses, log=print):
    """
    Execute a task plan (list of action dicts) using the robot.
    """
    for i, step in enumerate(plan):
        action = step['action']
        label = f'[Step {i+1}/{len(plan)}] {action}'
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

        arm.move_gohome(wait=True)
        time.sleep(0.5)

    log('Beverage preparation complete!')
    return True


def run_beverage_task(user_requirement='', confirm=True, log=print):
    """
    End-to-end pipeline: capture scene, plan, optionally confirm, execute.

    Parameters
    ----------
    user_requirement : str
        Free-form text describing the request (e.g. "coffee, lactose intolerant").
    confirm : bool
        If True, shows the captured image and waits for a keypress before executing.
        Set to False for GUI-driven flows that handle confirmation externally.
    log : callable
        Function used for progress messages (defaults to print). The GUI passes
        its own logger to route messages into the status area.

    Returns
    -------
    dict
        {"status": "ok"|"error", "message": str, "response": <raw API response>}
    """
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic
    detector = ContainerDetector(camera_intrinsic)

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
        cv_image = zed.image

        log('Detecting containers...')
        poses, poses_cam = detector.detect_all(cv_image)
        if poses is None or len(poses) == 0:
            return {'status': 'error', 'message': 'No containers detected.', 'response': None}
        log(f'Detected tag IDs: {list(poses.keys())}')

        # Draw pose axes on detected tags (for the confirmation window)
        for t_cam_obj in poses_cam.values():
            draw_pose_axes(cv_image, camera_intrinsic, t_cam_obj, size=CUBE_TAG_SIZE)

        log(f'User request: {user_requirement!r}')
        log('Sending image to OpenAI for task planning...')
        response = get_task_plan(cv_image, user_requirement)
        log('Received response:')
        log(json.dumps(response, indent=2))

        if response.get('status') != 'ok':
            message = response.get('message', 'Unknown planning error.')
            return {'status': 'error', 'message': message, 'response': response}

        plan = response.get('plan', [])
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
