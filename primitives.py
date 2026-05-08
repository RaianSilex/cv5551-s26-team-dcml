import cv2, numpy, time
from pupil_apriltags import Detector
from ultralytics import YOLO

from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, CUBE_TAG_FAMILY, CUBE_TAG_SIZE
from config import (
    INGREDIENT_TAG_MAP, STIRRER_TAG_ID, STIRRER_POSE_KEY, STIRRER_TAG_SIZE_M, STIRRER_GRASP_Z_OFFSET_MM, MAIN_CUP_POSITION,
    PRE_GRASP_HEIGHT, POUR_HEIGHT, STIR_DEPTH, STIR_CYCLES,
    POUR_TILT_ANGLE, STIR_CUP_POSITION, STIR_HEIGHT, CUP_RIM_Z_MM,
)


def _get_cup_xyz_mm():
    return (MAIN_CUP_POSITION['x'], MAIN_CUP_POSITION['y'], MAIN_CUP_POSITION['z'])

def _get_stir_cup_xyz_mm():
    return (STIR_CUP_POSITION['x'], STIR_CUP_POSITION['y'], STIR_CUP_POSITION['z'])


CUP_MODEL_PATH = '/home/rob/FPv3/cv5551-s26-team-dcml/best.pt'

CUP_CLASS_ALIASES = {
    'orange juice': 'orange',
}

CUP_RADIUS_M = 0.036
CUP_RADIUS_MM = 36.0

RIM_GRASP_OFFSET_MM = 5


class ContainerDetector:

    def __init__(self, camera_intrinsic):
        self.camera_intrinsic = camera_intrinsic

        print(f'Loading cup detection model from {CUP_MODEL_PATH}...')
        self.yolo = YOLO(CUP_MODEL_PATH)
        self.yolo.to('cpu')
        print(f'Model ready. Classes: {self.yolo.names}')

        self.apriltag_detector = Detector(
            families=CUBE_TAG_FAMILY,
            quad_decimate=1.0,
            decode_sharpening=0.5,
        )

    def detect_all(self, observation, depth_image=None, point_cloud=None):

        t_cam_robot = get_transform_camera_robot(observation, self.camera_intrinsic)
        if t_cam_robot is None:
            print('Could not compute camera-to-robot transform.')
            return None, None

        if len(observation.shape) > 2 and observation.shape[2] == 4:
            bgr = cv2.cvtColor(observation, cv2.COLOR_BGRA2BGR)
        else:
            bgr = observation

        fx = self.camera_intrinsic[0, 0]
        fy = self.camera_intrinsic[1, 1]
        cx = self.camera_intrinsic[0, 2]
        cy = self.camera_intrinsic[1, 2]

        poses_robot = {}
        poses_cam   = {}

        results = self.yolo.predict(bgr, conf=0.2, verbose=False)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:

                cls_name = self.yolo.names[int(box.cls[0])]
                ingredient_name = CUP_CLASS_ALIASES.get(cls_name, cls_name)
                conf = float(box.conf[0])                
                min_conf = 0.10 if ingredient_name == 'orange' else 0.2
                if conf < min_conf:
                    continue

                tag_id   = INGREDIENT_TAG_MAP.get(ingredient_name)
                if tag_id is None:
                    print(f'  Skipping class "{cls_name}" because it is not mapped to an ingredient.')
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                u_center = (x1 + x2) / 2.0
                v_rim = y1 + (y2 - y1) * 0.28
                cup_xyz_robot = _pixel_to_robot_on_plane(
                    u_center, v_rim, self.camera_intrinsic, t_cam_robot, CUP_RIM_Z_MM / 1000.0
                )
                if cup_xyz_robot is None:
                    print(f'  [{cls_name}] could not intersect camera ray with cup-height plane, skipping.')
                    continue

                x_robot, y_robot, z_robot = cup_xyz_robot
                print(
                    f'  [{cls_name}] plane-intersection xyz_robot='
                    f'({x_robot:.3f}, {y_robot:.3f}, {z_robot:.3f}) m'
                )

                xyz_cam_surface = _sample_cup_surface_point(point_cloud, x1, y1, x2, y2)
                if xyz_cam_surface is not None:
                    dbg_x, dbg_y, dbg_z = _surface_point_to_cup_center(*xyz_cam_surface)
                    print(f'  [{cls_name}] debug depth-center xyz_cam=({dbg_x:.3f}, {dbg_y:.3f}, {dbg_z:.3f}) m')

                t_robot_obj = numpy.eye(4)
                t_robot_obj[0, 3] = x_robot
                t_robot_obj[1, 3] = y_robot
                t_robot_obj[2, 3] = z_robot

                t_cam_obj = t_cam_robot @ t_robot_obj

                poses_robot[tag_id] = t_robot_obj
                poses_cam[tag_id]   = t_cam_obj

                print(f'  Detected {ingredient_name} from class {cls_name} (id={tag_id}) conf={conf:.2f} '
                      f'xy_robot=({t_robot_obj[0,3]*1000:.0f}, '
                      f'{t_robot_obj[1,3]*1000:.0f}) mm  '
                      f'z_fixed={CUP_RIM_Z_MM:.0f} mm')

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if len(bgr.shape) > 2 else bgr
        tags = self.apriltag_detector.detect(
            gray, estimate_tag_pose=True,
            camera_params=[fx, fy, cx, cy],
            tag_size=STIRRER_TAG_SIZE_M,
        )
        for tag in tags:
            if tag.tag_id != STIRRER_TAG_ID:
                continue
            t_cam_obj = numpy.eye(4)
            t_cam_obj[:3, :3] = tag.pose_R
            t_cam_obj[:3, 3]  = tag.pose_t.flatten()
            t_robot_obj = numpy.linalg.inv(t_cam_robot) @ t_cam_obj
            print(f'  [stirrer] robot xy = ({t_robot_obj[0,3]*1000:.0f}, {t_robot_obj[1,3]*1000:.0f}) mm')

            poses_robot[STIRRER_POSE_KEY] = t_robot_obj
            poses_cam[STIRRER_POSE_KEY]   = t_cam_obj
            print(f'  Detected stirrer (tag {STIRRER_TAG_ID}, stored as pose key {STIRRER_POSE_KEY})')

        return poses_robot, poses_cam


def _sample_depth(depth_image, u, v, x1, y1, x2, y2, fallback_z=0.5):
    """Median depth in a small patch around the bounding box center."""
    if depth_image is None:
        return fallback_z
    h, w = depth_image.shape[:2]
    u, v = int(u), int(v)
    ps   = 15
    patch = depth_image[max(0,v-ps):min(h,v+ps),
                        max(0,u-ps):min(w,u+ps)].astype(float)
    valid = patch[(patch > 0.1) & (patch < 5.0) & ~numpy.isnan(patch)]
    # if len(valid) == 0:
    #     return fallback_z
    return float(numpy.median(valid)) if len(valid) > 0 else fallback_z
    #return float(numpy.percentile(valid, 10))

def _sample_depth_wide(depth_image, x1, y1, x2, y2, fallback_z=0.65):
    if depth_image is None:
        return fallback_z
    h, w = depth_image.shape[:2]
    strip_y0 = max(0, int(y1 + (y2-y1) * 0.45))
    strip_y1 = min(h, int(y1 + (y2-y1) * 0.8))
    strip_x0 = max(0, x1 + 10)
    strip_x1 = min(w, x2 - 10)

    strip = depth_image[strip_y0:strip_y1,  strip_x0:strip_x1].astype(float)
    valid = strip[(strip > 0.1) & (strip < 5.0) & ~numpy.isnan(strip)]
    if len(valid) < 10:
        return fallback_z
    return float(numpy.median(valid))


def _sample_cup_surface_point(point_cloud, x1, y1, x2, y2):

    if point_cloud is None:
        return None

    h, w = point_cloud.shape[:2]
    strip_y0 = max(0, int(y1 + (y2 - y1) * 0.72))
    strip_y1 = min(h, int(y1 + (y2 - y1) * 0.92))
    strip_x0 = max(0, int(x1 + (x2 - x1) * 0.35))
    strip_x1 = min(w, int(x1 + (x2 - x1) * 0.65))

    if strip_y1 <= strip_y0 or strip_x1 <= strip_x0:
        return None

    xyz = point_cloud[strip_y0:strip_y1, strip_x0:strip_x1, :3].reshape(-1, 3).astype(float)
    valid = xyz[
        numpy.isfinite(xyz).all(axis=1)
        & (xyz[:, 2] > 0.1)
        & (xyz[:, 2] < 5.0)
    ]
    if len(valid) < 10:
        return None

    z_near = float(numpy.percentile(valid[:, 2], 20))
    close_to_surface = valid[numpy.abs(valid[:, 2] - z_near) < 0.03]
    if len(close_to_surface) >= 10:
        valid = close_to_surface

    median_xyz = numpy.median(valid, axis=0)
    return float(median_xyz[0]), float(median_xyz[1]), float(median_xyz[2])


def _surface_point_to_cup_center(x, y, z):
    ray = numpy.array([x, y, z], dtype=float)
    norm = numpy.linalg.norm(ray)
    if norm < 1e-6:
        return x, y, z
    center = ray + (ray / norm) * CUP_RADIUS_M
    return float(center[0]), float(center[1]), float(center[2])


def _pixel_to_robot_on_plane(u, v, camera_intrinsic, t_cam_robot, z_robot_plane):

    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]

    ray_cam = numpy.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
    ray_cam /= numpy.linalg.norm(ray_cam)

    t_robot_cam = numpy.linalg.inv(t_cam_robot)
    cam_origin_robot = t_robot_cam[:3, 3]
    ray_robot = t_robot_cam[:3, :3] @ ray_cam

    if abs(ray_robot[2]) < 1e-6:
        return None

    scale = (z_robot_plane - cam_origin_robot[2]) / ray_robot[2]
    if scale <= 0:
        return None

    point_robot = cam_origin_robot + scale * ray_robot
    return float(point_robot[0]), float(point_robot[1]), float(point_robot[2])


def _offset_grasp_to_rim_mm(center_x_mm, center_y_mm):

    toward_robot = numpy.array([-center_x_mm, -center_y_mm], dtype=float)
    norm = numpy.linalg.norm(toward_robot)
    if norm < 1e-6:
        return center_x_mm, center_y_mm
    offset = toward_robot / norm * CUP_RADIUS_MM
    return float(center_x_mm + offset[0]), float(center_y_mm + offset[1])



def pick_container(arm, container_pose):

    center_x = container_pose[0, 3] * 1000  
    center_y = container_pose[1, 3] * 1000
    x, y = _offset_grasp_to_rim_mm(center_x, center_y)
    print(f'  Container center (robot frame): x={center_x:.0f} mm, y={center_y:.0f} mm, '
          f'z={CUP_RIM_Z_MM:.0f} mm (fixed)')
    print(f'  Rim grasp target: x={x:.0f} mm, y={y:.0f} mm')

    grasp_z = CUP_RIM_Z_MM + RIM_GRASP_OFFSET_MM

    grasp_roll, grasp_pitch, grasp_yaw = 180, 0, 90

    arm.open_lite6_gripper()
    time.sleep(1.5)
    arm.stop_lite6_gripper()

    arm.set_position(x, y, grasp_z + PRE_GRASP_HEIGHT,
                     grasp_roll, grasp_pitch, grasp_yaw, wait=True)

    arm.set_position(x, y, grasp_z,
                     grasp_roll, grasp_pitch, grasp_yaw, wait=True)

    arm.close_lite6_gripper()
    time.sleep(0.8)

    arm.set_position(x, y, grasp_z + PRE_GRASP_HEIGHT,
                     grasp_roll, grasp_pitch, grasp_yaw, wait=True)


def place_container(arm, container_pose):
    center_x = container_pose[0, 3] * 1000
    center_y = container_pose[1, 3] * 1000
    x, y = _offset_grasp_to_rim_mm(center_x, center_y)
    grasp_z = CUP_RIM_Z_MM + RIM_GRASP_OFFSET_MM

    arm.set_position(x, y, grasp_z + PRE_GRASP_HEIGHT, 180, 0, 0, wait=True)
    arm.set_position(x, y, grasp_z, 180, 0, 0, wait=True)
    arm.open_lite6_gripper()
    time.sleep(1.5)
    arm.stop_lite6_gripper()
    arm.set_position(x, y, grasp_z + PRE_GRASP_HEIGHT, 180, 0, 0, wait=True)


def pick_stirrer(arm, stirrer_pose):

    x = stirrer_pose[0, 3] * 1000
    y = stirrer_pose[1, 3] * 1000 
    z = stirrer_pose[2, 3] * 1000 + 4

    x_axis = stirrer_pose[:3, 0]
    yaw = numpy.degrees(numpy.arctan2(x_axis[1], x_axis[0])) + 90
    grasp_roll, grasp_pitch, grasp_yaw = 180, 0, yaw
    grasp_z = z + STIRRER_GRASP_Z_OFFSET_MM

    print(f'  Stirrer pose (robot frame): x={x:.0f} mm, y={y:.0f} mm, z={z:.0f} mm, yaw={yaw:.1f} deg')
    print(f'  Stirrer grasp target z={grasp_z:.0f} mm (offset {STIRRER_GRASP_Z_OFFSET_MM:+.0f} mm from tag)')

    arm.open_lite6_gripper()
    time.sleep(1.5)
    arm.stop_lite6_gripper()

    arm.set_position(x, y, z + PRE_GRASP_HEIGHT, grasp_roll, grasp_pitch, grasp_yaw, wait=True)
    arm.set_position(x, y, grasp_z, grasp_roll, grasp_pitch, grasp_yaw, wait=True)
    arm.close_lite6_gripper()
    time.sleep(0.8)
    arm.set_position(x, y, z + PRE_GRASP_HEIGHT, grasp_roll, grasp_pitch, grasp_yaw, wait=True)


def place_stirrer(arm, stirrer_pose):
    x = stirrer_pose[0, 3] * 1000
    y = stirrer_pose[1, 3] * 1000
    z = stirrer_pose[2, 3] * 1000

    x_axis = stirrer_pose[:3, 0]
    yaw = numpy.degrees(numpy.arctan2(x_axis[1], x_axis[0])) + 90

    arm.set_position(x, y, z + PRE_GRASP_HEIGHT, 180, 0, yaw, wait=True)
    arm.set_position(x, y, z + STIRRER_GRASP_Z_OFFSET_MM, 180, 0, yaw, wait=True)
    arm.open_lite6_gripper()
    time.sleep(1.5)
    arm.stop_lite6_gripper()
    arm.set_position(x, y, z + PRE_GRASP_HEIGHT, 180, 0, yaw, wait=True)


def move_above_cup(arm):
    x, y, z = _get_cup_xyz_mm()
    arm.set_position(x, y, z + PRE_GRASP_HEIGHT + POUR_HEIGHT + 50, 180, 0, 180, wait=True)


def pour(arm):
    x, y, z  = _get_cup_xyz_mm()
    pour_z   = z + POUR_HEIGHT + 60
    angles   = [-61.3, 23, 67.5, 13, 42, 81.5]
    arm.set_servo_angle(angle=angles, speed = 50, wait=True)
    time.sleep(0.2)
    angles[4] = -65
    arm.set_servo_angle(angle=angles, speed = 80, wait=True)
    time.sleep(2.0)
    arm.set_position(x, y, pour_z, 180, -5, -46.8, wait=True)
    time.sleep(0.5)
    arm.set_position(x, y, pour_z + PRE_GRASP_HEIGHT, 180, 0, 0, wait=True)


def stir(arm):
    x, y, z  = _get_stir_cup_xyz_mm()
    top_z    = z + STIR_HEIGHT
    bottom_z = z - STIR_DEPTH
    arm.set_position(x, y, top_z + 50, 180, 0, 0, wait=True)
    arm.set_position(x, y, bottom_z, 180, 0, 0, wait=True)
    for _ in range(STIR_CYCLES):
        arm.set_position(x + 10, y, bottom_z, 180, 0, 0, wait=True)
        arm.set_position(x, y + 10, bottom_z, 180, 0, 0, wait=True)
        arm.set_position(x - 10, y, bottom_z, 180, 0, 0, wait=True)
        arm.set_position(x, y - 10, bottom_z, 180, 0, 0, wait=True)
    arm.set_position(x, y, bottom_z, 180, 0, 0, wait=True)
    arm.set_position(x, y, top_z + PRE_GRASP_HEIGHT, 180, 0, 0, wait=True)


def execute_add_ingredient(arm, ingredient_name, poses):
    tag_id = INGREDIENT_TAG_MAP[ingredient_name]
    if tag_id not in poses:
        print(f'[ERROR] "{ingredient_name}" not detected.')
        return False
    container_pose = poses[tag_id]
    print(f'  Picking up {ingredient_name}...')
    pick_container(arm, container_pose)
    print(f'  Moving above main cup...')
    move_above_cup(arm)
    print(f'  Pouring {ingredient_name}...')
    pour(arm)
    print(f'  Returning {ingredient_name} container...')
    place_container(arm, container_pose)
    return True


def execute_stir(arm, poses):
    if STIRRER_POSE_KEY not in poses:
        print(f'[ERROR] Stirrer (tag {STIRRER_TAG_ID}) not detected.')
        return False
    stirrer_pose = poses[STIRRER_POSE_KEY]
    print(f'  Picking up stirrer...')
    pick_stirrer(arm, stirrer_pose)
    print(f'  Stirring...')
    stir(arm)
    print(f'  Returning stirrer...')
    place_stirrer(arm, stirrer_pose)
    return True
