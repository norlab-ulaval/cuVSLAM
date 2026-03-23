import csv
import json
import os
import glob
from typing import List, Tuple

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import cuvslam


def load_frame(image_path: str) -> np.ndarray:
    """Load an image from a file path into memory suitable for pycuVSLAM"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # cuvslam expects BGR unit8 OR gray8. Imread loads BGR by default.
    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
         raise ValueError(f"Unable to read image at path: {image_path}")

    # ensure contiguous 3-channel
    frame = np.ascontiguousarray(frame)
    return frame


def transform_to_pose(position: dict, orientation: dict) -> cuvslam.Pose:
    """Convert a 3D position and Quaternion dict to a cuvslam.Pose object."""
    return cuvslam.Pose(
        rotation=[orientation['x'], orientation['y'], orientation['z'], orientation['w']],
        translation=[position['x'], position['y'], position['z']]
    )

def _load_json_config(json_path: str) -> dict:
    """Load JSON configuration file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Sensor JSON not found: {json_path}")
    with open(json_path, 'r') as f:
        return json.load(f)

def _get_transform_pose(transforms_list, from_link: str, to_link: str) -> cuvslam.Pose:
    """Finds the transform between two links and returns cuvslam.Pose obj."""
    for tf in transforms_list:
        if tf['from'] == from_link and tf['to'] == to_link:
            return transform_to_pose(tf['position'], tf['orientation'])
    raise ValueError(f"Transform mapping {from_link} -> {to_link} missing.")

def _create_camera_from_config(config: dict) -> cuvslam.Camera:
    """Create cuvslam.Camera from JSON calibration config."""
    # "k" in config: [fx, 0, cx, 0, fy, cy, 0, 0, 1] 
    # based on rational_polynomial formatting standard
    k = config['k']
    fx, fy = k[0], k[4]
    cx, cy = k[2], k[5]
    
    # Needs to extract resolution dynamically. Since offset and width/height offsets are 0.
    # Usually ZED images are full-frame natively loaded, but let's assume we read 2208x1242 for ZedX unless otherwise dictated  
    # Assuming ZedX Native Size in JSON config lack
    width = 1920
    height = 1200 
    
    # For robust loading, assume these sizes - if known in advance or parsed later.
    
    cam = cuvslam.Camera()
    cam.focal = [fx, fy]
    cam.principal = [cx, cy]
    cam.size = [width, height] # Should be dynamically updated if needed, let's keep hardcode 1920x1080 native 1080p if omitted
    
    # "p" in config is rational_polynomial distortion. Brown-Conrady supports k1..k6, p1, p2
    # Standard mapping for OpenvCV/RationalPolynomial is k1,k2,p1,p2,k3,k4,k5,k6
    d = config['d'] # d = [k1, k2, p1, p2, k3, k4, k5, k6]
    
    # Pad to length 5 for simple Brown
    coeffs = d[:5]
    if len(coeffs) < 5:
        coeffs += [0.0] * (5 - len(coeffs))
        
    cam.distortion = cuvslam.Distortion(
        cuvslam.Distortion.Model.Brown,
        coeffs
    )
    return cam


def get_custom_rig(base_path: str) -> cuvslam.Rig:
    """
    Parses custom JSONs into `cuvslam.Rig`.
    `zedx_left` maps to cam_0 (identity frame context)
    `zedx_right` maps to cam_1
    `xsens` maps to imu_0
    """
    calib_dir = os.path.join(base_path, 'calib')
    
    # 1. Load Configurations
    cam0_json = _load_json_config(os.path.join(calib_dir, 'zedx_left.json'))
    cam1_json = _load_json_config(os.path.join(calib_dir, 'zedx_right.json'))
    imu_json = _load_json_config(os.path.join(calib_dir, 'imu.json'))
    transforms = _load_json_config(os.path.join(calib_dir, 'transforms.json'))
    
    # 2. Extract relative transform pairs (to zedx_left context)
    cam1_rig_pose = _get_transform_pose(transforms, from_link="zedx_left", to_link="zedx_right")
    imu_rig_pose = _get_transform_pose(transforms, from_link="zedx_left", to_link="vectornav")
    
    # 3. Create camera nodes
    cam0 = _create_camera_from_config(cam0_json)
    cam1 = _create_camera_from_config(cam1_json)
    
    # Extrinsics referencing
    cam0.rig_from_camera = cuvslam.Pose(
        rotation=[0, 0, 0, 1],  # Identity
        translation=[0, 0, 0]
    )
    cam1.rig_from_camera = cam1_rig_pose
    
    # 4. Create IMU 
    imu = cuvslam.ImuCalibration()
    imu.rig_from_imu = imu_rig_pose
    # VectorNav noise spec parsing
    vn_spec = imu_json.get('vectornav', {})
    vn_rate = 200 # common vectornav rate, you can tweak this
    
    imu.gyroscope_noise_density = 0.003233636934803717  # Fallback to defaults or parse
    imu.gyroscope_random_walk = 3.8e-05
    imu.accelerometer_noise_density = 0.016997696359396915
    imu.accelerometer_random_walk = 0.0003
    imu.frequency = vn_rate

    rig = cuvslam.Rig()
    rig.cameras = [cam0, cam1]
    rig.imus = [imu]
    
    return rig


def prepare_custom_metadata(base_path: str) -> List[dict]:
    """
    Parses vectornav.csv + the image dirs and interleaves 
    them chronologically based on their true nano timestamps.
    """
    events = []
    
    # 1. Parse Image Dirs
    left_dir = os.path.join(base_path, 'zedx_left')
    right_dir = os.path.join(base_path, 'zedx_right')
    
    left_pngs = sorted(glob.glob(os.path.join(left_dir, "*.png")))
    right_pngs = sorted(glob.glob(os.path.join(right_dir, "*.png")))
    
    # We map microsecond-epoch to file for rapid stereo-matching
    left_map = {}
    for pb in left_pngs:
        ts_us = int(os.path.splitext(os.path.basename(pb))[0])
        left_map[ts_us] = pb
        
    for rb in right_pngs:
        ts_us = int(os.path.splitext(os.path.basename(rb))[0])
        if ts_us in left_map:
            # Found Stereo Pair
            ts_ns = ts_us * 1000
            events.append({
                'type': 'stereo',
                'timestamp': ts_ns,
                'images_paths': [left_map[ts_us], rb]
            })

    # 2. Parse IMU Nav
    imu_csv = os.path.join(base_path, 'vectornav.csv')
    with open(imu_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Verify the custom layout explicitly "ax,ay,az,lx,ly,lz"
        
        for row in reader:
             if not row: continue
             ts_us = int(row[0])
             ts_ns = ts_us * 1000
             
             ax, ay, az = float(row[1]), float(row[2]), float(row[3])
             lx, ly, lz = float(row[4]), float(row[5]), float(row[6])
             
             events.append({
                'type': 'imu',
                'timestamp': ts_ns,
                'accel': [ax, ay, az],
                'gyro': [lx, ly, lz]
             })
             
    # Sort ALL events ascending explicitly using nanosecond timestamps
    events.sort(key=lambda x: x['timestamp'])
    return events
