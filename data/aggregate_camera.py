import os
import json


if __name__ == '__main__':

    src_camera_root_path = '/ssd/ray3d/Train/json'
    dst_camera_root_path = '/ssd/ray3d/training.json'

    valid_cameras = sorted(os.listdir(src_camera_root_path))
    src_camera_data = list()
    for camera in valid_cameras:
        meta = json.load(open(os.path.join(src_camera_root_path, camera), 'r'))[0]
        src_camera_data.append(meta)
    print('number of training {}'.format(len(valid_cameras)))
    json.dump(src_camera_data, open(dst_camera_root_path, 'w'), indent=4)

    # --------------------------------

    src_camera_root_path = [
        '/ssd/ray3d/Pitch/json',
        '/ssd/ray3d/Rotation/json',
        '/ssd/ray3d/Translation/json',
    ]
    dst_camera_root_path = '/ssd/ray3d/testing.json'

    camera_set = set()
    valid_cameras = list()
    for root_path in src_camera_root_path:
        current_cameras = sorted(os.listdir(root_path))
        for cam in current_cameras:
            if cam in camera_set:
                print(cam)
                continue
            else:
                camera_set.add(cam)
                valid_cameras.append(os.path.join(root_path, cam))
    print('number of testing {}'.format(len(valid_cameras)))
    src_camera_data = list()
    for camera in valid_cameras:
        meta = json.load(open(camera, 'r'))[0]
        src_camera_data.append(meta)

    json.dump(src_camera_data, open(dst_camera_root_path, 'w'), indent=4)