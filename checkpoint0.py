import cv2, numpy
from pupil_apriltags import Detector

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera

TAG_SIZE = 0.08

TAG_CENTER_COORDINATES = [[0.38, 0.4],
                         [0.38, -0.4],
                         [0.0, 0.4],
                         [0.0, -0.4]]

def get_pnp_pairs(tags):

    world_points = numpy.empty([0, 3])
    image_points = numpy.empty([0, 2])

    for tag in tags:
        
        if tag.tag_id > 3:
            continue
        
        tag_center = TAG_CENTER_COORDINATES[tag.tag_id]

        wp = numpy.zeros(3)
        wp[0] = tag_center[0] - (TAG_SIZE / 2)
        wp[1] = tag_center[1] + (TAG_SIZE / 2)

        ip = tag.corners[0]

        world_points = numpy.vstack([world_points, wp])
        image_points = numpy.vstack([image_points, ip])

        wp = numpy.zeros(3)
        wp[0] = tag_center[0] - (TAG_SIZE / 2)
        wp[1] = tag_center[1] - (TAG_SIZE / 2)

        ip = tag.corners[1]

        world_points = numpy.vstack([world_points, wp])
        image_points = numpy.vstack([image_points, ip])

        wp = numpy.zeros(3)
        wp[0] = tag_center[0] + (TAG_SIZE / 2)
        wp[1] = tag_center[1] - (TAG_SIZE / 2)

        ip = tag.corners[2]

        world_points = numpy.vstack([world_points, wp])
        image_points = numpy.vstack([image_points, ip])

        wp = numpy.zeros(3)
        wp[0] = tag_center[0] + (TAG_SIZE / 2)
        wp[1] = tag_center[1] + (TAG_SIZE / 2)

        ip = tag.corners[3]

        world_points = numpy.vstack([world_points, wp])
        image_points = numpy.vstack([image_points, ip])

    return world_points, image_points

def get_transform_camera_robot(observation, camera_intrinsic):

    detector = Detector(families='tag36h11')

    if len(observation.shape) > 2:
        observation = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
    tags = detector.detect(observation, estimate_tag_pose=False)
    print(f'Number of tags found: {len(tags)}')
    world_points, image_points = get_pnp_pairs(tags)
    if world_points.shape[0] < 4:
        print(f'Insufficient valid tag corners found.')
        return None

    success, rotation_vec, translation = cv2.solvePnP(world_points, image_points, camera_intrinsic, None)
    if success is not True:
        print('PnP Calculation Failed.')
        return None
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    transform_mat = numpy.eye(4)
    transform_mat[:3, :3] = rotation_mat
    transform_mat[:3, 3] = translation.flatten()

    return transform_mat

def main():

    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    try:
        cv_image = zed.image

        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return
        
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_robot, size=TAG_SIZE)
        cv2.namedWindow('Verifying World Origin', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying World Origin', 1280, 720)
        cv2.imshow('Verifying World Origin', cv_image)
        cv2.waitKey(0)
    
    finally:
        zed.close()

if __name__ == "__main__":
    main()
