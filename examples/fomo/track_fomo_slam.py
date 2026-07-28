# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

import os
import sys
import json
import argparse
import threading
import time
import csv
import queue
import concurrent.futures
import numpy as np
from PIL import Image
from numpy import loadtxt, asarray, array_equal as np_array_equal, savetxt
from scipy.spatial.transform import Rotation as R
import rerun as rr
import rerun.blueprint as rrb
import cuvslam

from fomo_sdk.tf.utils import FoMoTFTree
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description="Track FOMO dataset sequence with SLAM")
parser.add_argument("--sequence_dir", type=str, required=True, help="Path to the sequence directory")
parser.add_argument("--slam_sync_mode", action="store_true", help="Enable sync slam thread")
parser.add_argument("--idx", type=int, default=0, help="Starting index of the sequence after localization. If negative, don't localize but run SLAM and map.")
parser.add_argument("--max_wait_time", type=float, default=10.0, help="Max wait time in seconds")
parser.add_argument("--output_filepath", type=str, default="", help="Output filepath. If empty, don't save.")
parser.add_argument("--no_vis", action="store_true", help="Disable rerun visualization")
parser.add_argument("--no_slam", action="store_true", help="Disable SLAM (run Odometry only)")
parser.add_argument("--no_imu", action="store_true", help="Disable IMU (run Visual Odometry only)")
parser.add_argument("--localize", action="store_true", help="Enable localization using existing map. Otherwise runs SLAM and mapping.")
args = parser.parse_args()

args.no_imu = True

if args.output_filepath:
    log_dir = args.output_filepath
    if args.localize:
        seq_name = os.path.basename(os.path.normpath(args.sequence_dir))
        map_path_test = os.path.join(args.output_filepath, 'map')
        traj_test = os.path.join(args.output_filepath, 'trajectory_tum.txt')
        if os.path.exists(map_path_test) and os.path.exists(traj_test) and args.idx >= 0:
            log_dir = os.path.join(args.output_filepath, "loc_" + seq_name)
    
    os.makedirs(log_dir, exist_ok=True)
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    sys.stdout = Logger(os.path.join(log_dir, "log.txt"))

# Dataset sequence to track and visualize
sequence_path = os.path.abspath(args.sequence_dir)
calib_path = os.path.join(sequence_path, "calib")

# Lambda to convert quaternion [x, y, z, w] to 3x3 rotation matrix (as list of lists)
quaternion_to_rotation_matrix = lambda q: R.from_quat(q).as_matrix().tolist()

# Lambda to multiply two quaternions [x, y, z, w] * [x, y, z, w]
quaternion_multiply = lambda q1, q2: (R.from_quat(q1) * R.from_quat(q2)).as_quat()

# Lambda to rotate a 3D vector using a 3x3 rotation matrix
rotate_vector = lambda vector, rotation_matrix: R.from_matrix(rotation_matrix).apply(vector)

tf_tree = FoMoTFTree()


def transform_to_pose(transform_matrix: np.ndarray) -> cuvslam.Pose:
    """Convert a 4x4 transformation matrix to a cuvslam.Pose object."""
    rotation_quat = Rotation.from_matrix(transform_matrix[:3, :3]).as_quat()
    return cuvslam.Pose(rotation=rotation_quat, translation=transform_matrix[:3, 3])

class Noise():
    def __init__(self, gnd, grw, acnd, acrw):
        self.gnd = gnd
        self.grw = grw
        self.acnd = acnd
        self.acrw = acrw

def get_imu_noise(file_path: str) -> Noise:    
    imu_data = {
        "accelerometer": {},
        "gyroscope": {}
    }
    
    current_sensor = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Determine which sensor block we are in
            if line.startswith("ACCELEROMETER:"):
                current_sensor = "accelerometer"
                continue
            elif line.startswith("GYROSCOPE:"):
                current_sensor = "gyroscope"
                continue
                
            # Parse the key-value pairs if we are inside a sensor block
            if current_sensor and ':' in line:
                key, values_str = line.split(':', 1)
                key = key.strip()
                
                # Split the remaining string by whitespace and grab the first token
                tokens = values_str.split()
                if tokens:
                    try:
                        # The first token is always the primary numerical value
                        val = float(tokens[0])
                        imu_data[current_sensor][key] = val
                    except ValueError:
                        # Skip lines that don't have a parseable float first
                        continue
    # print(imu_data)

    acc = imu_data["accelerometer"]
    # nd = ""
    rw = "Accel Random Walk"
    # acnd = 
    acrw = np.linalg.norm(np.array([acc["X "+ rw], acc["Y "+ rw], acc["Z "+ rw]]))


    # same values as orbslam3
    # IMU.NoiseGyro: 6.170024194584988e-5 # 2.44e-4 #1e-3 # rad/s^0.5
    # IMU.NoiseAcc: 0.0014475547298752243 # 1.47e-3 #1e-2 # m/s^1.5
    # IMU.GyroWalk: 1.2754866476100412e-6 # rad/s^1.5
    # IMU.AccWalk: 1.743080431796351e-5 # m/s^2.5

    acnd = 0.0014475547298752243
    acrw = 1.743080431796351e-5
    gnd = 6.170024194584988e-05
    grw = 1.2754866476100412e-06
    return Noise(gnd, grw, acnd, acrw)


def combine_poses(initial_pose, relative_pose):
    # Get rotation matrix from initial pose quaternion
    rotation_matrix = quaternion_to_rotation_matrix(initial_pose.rotation)

    # Rotate relative translation by initial pose rotation
    rotated_rel_t = rotate_vector(relative_pose.translation, rotation_matrix)

    # Add initial translation
    absolute_translation = [
        initial_pose.translation[0] + rotated_rel_t[0],
        initial_pose.translation[1] + rotated_rel_t[1],
        initial_pose.translation[2] + rotated_rel_t[2]
    ]

    # Multiply quaternions
    absolute_rotation = quaternion_multiply(initial_pose.rotation, relative_pose.rotation)

    return cuvslam.Pose(translation=absolute_translation, rotation=absolute_rotation)


def transform_landmarks(landmarks, initial_pose):
    if not landmarks:
        return []
    rotation_matrix = quaternion_to_rotation_matrix(initial_pose.rotation)
    rot = R.from_matrix(rotation_matrix)
    
    landmarks_np = np.array(landmarks)
    rotated_landmarks = rot.apply(landmarks_np)
    
    trans = np.array(initial_pose.translation)
    transformed_landmarks = rotated_landmarks + trans
    
    return transformed_landmarks.tolist()


def save_callback(success):
    global map_saved
    map_saved = success

def localization_start_cb():
    print("Localization started")

def localization_finish_cb(pose, error_message):
    global slam_initial_pose
    if pose is not None:
        print(f"[Localization] Map loaded and localized successfully! Initial translation: {pose.translation}")
    else:
        print(f"[Localization] Failed to localize: {error_message}")
    slam_initial_pose = pose
    localization_complete.set()

def color_from_id(identifier):
    return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]

if not args.no_vis:
    rr.init('fomo', strict=True, spawn=True)

    rr.send_blueprint(rrb.Blueprint(
        rrb.TimePanel(state="collapsed"),
        rrb.Horizontal(
            column_shares=[0.5, 0.5],
            contents=[
                rrb.Vertical(contents=[
                    rrb.Spatial2DView(origin='world/car/cam0'),
                    rrb.Vertical(contents=[
                        rrb.TimeSeriesView(
                        name="IMU Acceleration",
                        origin="world/imu/accel",
                        overrides={
                            "world/imu/accel/x": rr.SeriesLine.from_fields(color=[255, 0, 0]),
                            "world/imu/accel/y": rr.SeriesLine.from_fields(color=[0, 255, 0]),
                            "world/imu/accel/z": rr.SeriesLine.from_fields(color=[0, 0, 255]),
                        },
                    ),
                    rrb.TimeSeriesView(
                        name="IMU Angular Velocity",
                        origin="world/imu/gyro",
                        overrides={
                            "world/imu/gyro/x": rr.SeriesLine.from_fields(color=[255, 0, 0]),
                            "world/imu/gyro/y": rr.SeriesLine.from_fields(color=[0, 255, 0]),
                            "world/imu/gyro/z": rr.SeriesLine.from_fields(color=[0, 0, 255]),
                        },
                    )
                    ])
                ]),
                rrb.Spatial3DView(origin='world')
            ]
        )
    ))

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    rr.log("world/xyz", rr.Arrows3D(
        vectors=[[20, 0, 0], [0, 20, 0], [0, 0, 20]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        labels=['[x]', '[y]', '[z]']
    ), static=True)

SLAM_SYNC_MODE = False
IDX = args.idx
max_wait_time = args.max_wait_time
USE_SLAM = not args.no_slam
USE_IMU = not args.no_imu

with open(os.path.join(calib_path, 'transforms.json'), 'r') as f:
    transforms = json.load(f)

cameras = [cuvslam.Camera(), cuvslam.Camera()]

# zedx left to base_link
tf_zedx_left_to_base_link = tf_tree.get_transform(from_frame="base_link", to_frame="zedx_left")

# remove the rotation before right and left zedx lenses
tf_zedx_right_to_base_link = tf_zedx_left_to_base_link

tf_zedx_right_to_zedx_left = tf_tree.get_transform(from_frame="zedx_left", to_frame="zedx_right")

translation_only = np.eye(4)
translation_only[:3, 3] = tf_zedx_right_to_zedx_left[:3, 3]

tf_zedx_right_to_base_link = tf_zedx_left_to_base_link @ translation_only

cameras[0].rig_from_camera = transform_to_pose(tf_zedx_left_to_base_link)
cameras[1].rig_from_camera = transform_to_pose(tf_zedx_right_to_base_link)

left_dir = os.path.join(sequence_path, "zedx_left")
right_dir = os.path.join(sequence_path, "zedx_right")

filenames = sorted(os.listdir(left_dir))
size = Image.open(os.path.join(left_dir, filenames[0])).size

with open(os.path.join(calib_path, 'zedx_left.json'), 'r') as f:
    left_intrinsics = json.load(f)
with open(os.path.join(calib_path, 'zedx_right.json'), 'r') as f:
    right_intrinsics = json.load(f)

for i, intrinsics in enumerate([left_intrinsics, right_intrinsics]):
    cameras[i].size = size
    cameras[i].focal = [intrinsics["k"][0], intrinsics["k"][4]]
    cameras[i].principal = [intrinsics["k"][2], intrinsics["k"][5]]

# IMU Configuration
if USE_IMU:
    # vectornav to base_link
    # tf_vectornav_to_base_link = tf_tree.get_transform(from_frame="vectornav", to_frame="base_link")
    tf_vectornav_to_base_link = tf_tree.get_transform(from_frame="base_link", to_frame="vectornav")

    imu = cuvslam.ImuCalibration()
    imu.rig_from_imu = transform_to_pose(tf_vectornav_to_base_link)

    noise = get_imu_noise(os.path.join(calib_path, 'allan-vectornav.txt'))

    imu.gyroscope_noise_density = noise.gnd
    imu.gyroscope_random_walk = noise.grw
    imu.accelerometer_noise_density = noise.acnd
    imu.accelerometer_random_walk = noise.acrw
    imu.frequency = 200.0
else:
    print("Not using IMU")

rig = cuvslam.Rig()
rig.cameras = cameras
if USE_IMU:
    rig.imus = [imu]

odometry_mode = cuvslam.Tracker.OdometryMode.Inertial if USE_IMU else cuvslam.Tracker.OdometryMode.Multicamera

cfg = cuvslam.Tracker.OdometryConfig(
    async_sba=False,
    enable_final_landmarks_export=True,
    rectified_stereo_camera=True,
    odometry_mode=odometry_mode
)

if USE_SLAM:
    # Use async SLAM (sync_mode=False) even during localization.
    # With sync_mode=True, localize_in_map() itself blocks until done, which
    # freezes the script for large maps. In async mode, localize_in_map()
    # returns immediately and the blocking frame-feeding loop below drives
    # the localization to completion via incremental tracker.track() calls.
    s_cfg = cuvslam.Tracker.SlamConfig(sync_mode=SLAM_SYNC_MODE, enable_mapping=not args.localize)
    tracker = cuvslam.Tracker(rig, cfg, s_cfg)
else:
    tracker = cuvslam.Tracker(rig, cfg)

timestamps = [int(os.path.splitext(f)[0]) * 1000 for f in filenames]

# Check if map folder and trajectory file exist
# map/ is used directly for both saving (mapping run) and loading (localization).
# This matches track_fomo_kitti_slam.py: no rename or copy is needed.
map_path = os.path.join(args.output_filepath, 'map')
trajectory_file = os.path.join(args.output_filepath, 'trajectory_tum.txt')

if not os.path.exists(map_path) and args.localize:
    print(f"Map folder not found at {map_path} — localization will not run.")
elif not os.path.exists(map_path):
    print(f"Map folder not found at {map_path} (will be created during mapping).")

localization_complete = threading.Event()
slam_initial_pose = None
guess_pose = None
map_saved = False
total_db_landmarks = 0

loc_settings = cuvslam.Tracker.SlamLocalizationSettings(
    horizontal_search_radius=8.,
    vertical_search_radius=2.,
    horizontal_step=0.5,
    vertical_step=0.2,
    angular_step_rads=0.03
    )

if not args.localize:
    guess_pose = None
    print("Localization flag not set, skipping localization and running SLAM mapping.")
elif IDX < 0:
    IDX = 0
    guess_pose = None
    print("IDX is negative, skipping localization and starting SLAM from frame 0.")
elif os.path.exists(trajectory_file) and os.path.exists(map_path):
    trajectory_data = loadtxt(trajectory_file)
    if IDX >= len(trajectory_data):
        raise IndexError(
            f"IDX ({IDX}) is out of bounds for loaded trajectory of length {len(trajectory_data)}"
        )
    guess_tum_pose = trajectory_data[IDX]
    guess_pose = cuvslam.Pose(translation=guess_tum_pose[1:4], rotation=guess_tum_pose[4:])

start_timestamp_ns = timestamps[IDX] if IDX < len(timestamps) else 0

# Prepare unified frames metadata (IMU and Cameras)
frames_metadata = []

for f_name in filenames:
    t_ns = int(os.path.splitext(f_name)[0]) * 1000
    frames_metadata.append({
        'type': 'camera',
        'timestamp': t_ns,
        'images_paths': [
            os.path.join(left_dir, f_name),
            os.path.join(right_dir, f_name)
        ]
    })

imu_csv_path = os.path.join(sequence_path, 'vectornav.csv')
imu_json_path = os.path.join(calib_path, 'imu.json')
with open(imu_json_path, 'r') as f:
    imu_biases = json.load(f)
gyro_bias = imu_biases.get('vectornav', {}).get('angular_velocity', {'x': 0, 'y': 0, 'z': 0})
bx, by, bz = gyro_bias['x'], gyro_bias['y'], gyro_bias['z']

with open(imu_csv_path, 'r') as f:
    next(f)
    reader = csv.reader(f)
    for row in reader:
        if not row: continue
        t_us = int(row[0])
        t_ns = t_us * 1000
        ax, ay, az = float(row[1]), float(row[2]), float(row[3])
        lx, ly, lz = float(row[4]), float(row[5]), float(row[6])
        frames_metadata.append({
            'type': 'imu',
            'timestamp': t_ns,
            'gyro': [ax - bx, ay - by, az - bz],
            'accel': [lx, ly, lz]
        })

frames_metadata.sort(key=lambda x: x['timestamp'])

processed_start_frame = False
# Localize in map if applicable.
# Mirrors track_fomo_kitti_slam.py: cuVSLAM localization runs fully in the
# background. No tracker.track() calls are needed to drive it — just wait.
# Large outdoor maps (thousands of landmarks) need more than the default 10s,
# so the timeout is max_wait_time * 10 (100s by default).
if USE_SLAM and os.path.exists(map_path) and (guess_pose is not None):
    init_images = None
    for meta in frames_metadata:
        if meta['type'] == 'camera' and meta['timestamp'] == start_timestamp_ns:
            init_images = [
                asarray(Image.open(meta['images_paths'][0]).convert('L')),
                asarray(Image.open(meta['images_paths'][1]).convert('L'))
            ]
            break

    if init_images is not None:
        if not args.no_vis:
            rr.set_time_nanos('timestamp', start_timestamp_ns)
            rr.log('world/car/cam0/image', rr.Image(init_images[0]))

        # Seed frame: track once, then kick off async localization
        processed_start_frame = True
        _, _ = tracker.track(start_timestamp_ns, init_images)
        tracker.localize_in_map(map_path, start_timestamp_ns, guess_pose, init_images,
                                loc_settings, localization_start_cb, localization_finish_cb)

        # Wait passively for the async localization callback to fire.
        # The search is CPU/GPU-bound; no additional track() calls are needed.
        loc_timeout_s = max_wait_time * 10  # generous: 100s by default
        wait_time = 0.0
        while not SLAM_SYNC_MODE and not localization_complete.wait(timeout=0.5) and wait_time < loc_timeout_s:
            print(f"Waiting for localization... elapsed: {wait_time:.0f}s / {loc_timeout_s:.0f}s")
            wait_time += 0.5
        if not localization_complete.is_set():
            print(f"[Localization] Did not complete within {loc_timeout_s:.0f}s.")
            exit(1)

# Determine fast-forward logic
skip_until_ns = start_timestamp_ns if processed_start_frame else -1
if slam_initial_pose is not None and guess_pose is not None:
    print(f"Localized pose: {slam_initial_pose}")
else:
    if args.localize and guess_pose is None:
        import sys
        print("Error: --localize flag is set, but guess_pose is None (trajectory missing). Exiting.")
        sys.exit(1)
    print("Warning: slam_initial_pose is None (or localization failed), setting initial tracking origin to zero.")
    slam_initial_pose = cuvslam.Pose(translation=[0, 0, 0], rotation=[0, 0, 0, 1])

trajectory = []
trajectory_slam = []
trajectory_tum = []
trajectory_odom_tum = []
loop_closure_poses = []
initial_map_size = None
loop_closures_log = []
timing_logs = []

executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

def load_images(image_paths):
    return [asarray(Image.open(p).convert('L')) for p in image_paths]

# Bounded queue to preload/preprocess input images (load 100 images)
image_queue = queue.Queue(maxsize=100)

camera_frames = [meta for meta in frames_metadata if meta['type'] == 'camera']

def image_producer():
    try:
        for meta in camera_frames:
            future = executor.submit(load_images, meta['images_paths'])
            image_queue.put((meta['timestamp'], future))
    except Exception:
        pass

producer_thread = threading.Thread(target=image_producer, daemon=True)
producer_thread.start()

frame_idx = 0

last_camera_frame_time = time.perf_counter()
total_sleep_time_since_last_cam = 0.0

# Main tracking loop
for metadata in frames_metadata:
    t_sleep_start = time.perf_counter()
    if metadata['type'] != 'imu':
        time.sleep(0.01)  # Give the async SLAM thread time to catch up
    total_sleep_time_since_last_cam += (time.perf_counter() - t_sleep_start)
    timestamp = metadata['timestamp']

    if timestamp <= skip_until_ns:
        if metadata['type'] == 'camera':
            image_queue.get()
            frame_idx += 1
        continue

    if metadata['type'] == 'imu':
        if not USE_IMU:
            continue

        imu_measurement = cuvslam.ImuMeasurement()
        imu_measurement.timestamp_ns = timestamp
        imu_measurement.linear_accelerations = np.asarray(metadata['accel'])
        imu_measurement.angular_velocities = np.asarray(metadata['gyro'])
        tracker.register_imu_measurement(0, imu_measurement)

        if not args.no_vis:
            rr.set_time_nanos('timestamp', timestamp)
            rr.log("world/imu/accel/x", rr.Scalar(metadata['accel'][0]), static=False)
            rr.log("world/imu/accel/y", rr.Scalar(metadata['accel'][1]), static=False)
            rr.log("world/imu/accel/z", rr.Scalar(metadata['accel'][2]), static=False)
            rr.log("world/imu/gyro/x", rr.Scalar(metadata['gyro'][0]), static=False)
            rr.log("world/imu/gyro/y", rr.Scalar(metadata['gyro'][1]), static=False)
            rr.log("world/imu/gyro/z", rr.Scalar(metadata['gyro'][2]), static=False)
        continue

    # Camera frame
    t_frame_start = time.perf_counter()
    q_timestamp, image_future = image_queue.get()
    assert q_timestamp == timestamp, f"Timestamp mismatch: queue {q_timestamp} vs frame {timestamp}"
    images = image_future.result()
    t_img_end = time.perf_counter()

    odometry_pose_estimate = tracker.odom.track(timestamp, images, None, None, None)
    t_odom_end = time.perf_counter()
    
    slam_pose = None
    if USE_SLAM and hasattr(tracker, 'slam') and tracker.slam and odometry_pose_estimate.world_from_rig:
        state = tracker.odom.get_state()
        slam_pose = tracker.slam.track(state, None)
    t_slam_end = time.perf_counter()
    
    img_time_ms = (t_img_end - t_frame_start) * 1000.0
    odom_time_ms = (t_odom_end - t_img_end) * 1000.0
    slam_time_ms = (t_slam_end - t_odom_end) * 1000.0

    timing_logs.append({
        'frame': frame_idx,
        'timestamp': f"{(timestamp / 1_000_000_000.0):.6f}",
        'odom_time_ms': odom_time_ms,
        'slam_time_ms': slam_time_ms,
        'total_time_ms': odom_time_ms + slam_time_ms,
        'full_frame_time_ms': 0.0,
        'viz_time_ms': 0.0,
        'loop_time_ms': 0.0,
        'sleep_time_ms': 0.0
    })

    if odometry_pose_estimate.world_from_rig is None:
        timing_logs[-1]['full_frame_time_ms'] = (time.perf_counter() - t_frame_start) * 1000.0
        print(f"Warning: Failed to track frame {frame_idx}")
        frame_idx += 1
        continue

    odom_pose = odometry_pose_estimate.world_from_rig.pose
    current_pose = combine_poses(slam_initial_pose, odom_pose)

    t_fetch_start = time.perf_counter()
    observations = tracker.get_last_observations(0)
    landmarks = tracker.get_last_landmarks()
    
    needs_viz = not args.no_vis
    needs_print = (frame_idx % 50 == 0 or frame_idx == len(timestamps) - 1)
    needs_map_size = needs_viz or needs_print or (initial_map_size is None)
    
    num_final_landmarks = 0
    final_landmarks = []
    
    if needs_viz:
        raw_final_landmarks = list(tracker.get_final_landmarks().values())
        t_fetch_end = time.perf_counter()
        final_landmarks = transform_landmarks(raw_final_landmarks, slam_initial_pose)
        num_final_landmarks = len(final_landmarks)
        t_transform_end = time.perf_counter()
        fetch_time_ms = (t_fetch_end - t_fetch_start) * 1000.0
        transform_time_ms = (t_transform_end - t_fetch_end) * 1000.0
    elif needs_map_size:
        num_final_landmarks = len(tracker.get_final_landmarks())
        t_fetch_end = time.perf_counter()
        fetch_time_ms = (t_fetch_end - t_fetch_start) * 1000.0
        transform_time_ms = 0.0
    else:
        fetch_time_ms = (time.perf_counter() - t_fetch_start) * 1000.0
        transform_time_ms = 0.0
        
    gravity = tracker.get_last_gravity() if USE_IMU else None

    if needs_viz:
        observations_uv = [[o.u, o.v] for o in observations]
        observations_colors = [color_from_id(o.id) for o in observations]
        landmark_xyz = [l.coords for l in landmarks]
        landmarks_colors = [color_from_id(l.id) for l in landmarks]

    trajectory.append(current_pose.translation)
    if USE_SLAM and slam_pose is not None:
        # Only record SLAM trajectory after localization has confirmed a pose.
        # (In mapping mode localization_complete is never set, so the condition
        # evaluates to True via the `not args.localize` branch.)
        if not args.localize or localization_complete.is_set():
            trajectory_slam.append(slam_pose.translation)
            trajectory_tum.append([timestamp / 1_000_000_000.0] + list(slam_pose.translation) + list(slam_pose.rotation))
    trajectory_odom_tum.append([timestamp / 1_000_000_000.0] + list(current_pose.translation) + list(current_pose.rotation))

    if USE_SLAM and hasattr(tracker, 'get_loop_closure_poses'):
        current_lc_poses = tracker.get_loop_closure_poses()
        if current_lc_poses:
            newest_lc = current_lc_poses[-1]
            if not loop_closures_log or newest_lc.timestamp_ns != loop_closures_log[-1]['timestamp_ns']:
                print(f"[Loop Closure] Detected new loop closure at frame {frame_idx} (timestamp: {newest_lc.timestamp_ns})")
                loop_closures_log.append({
                    'timestamp_ns': newest_lc.timestamp_ns,
                    'translation': [float(x) for x in newest_lc.pose.translation],
                    'rotation': [float(x) for x in newest_lc.pose.rotation]
                })
                loop_closure_poses.append(newest_lc.pose.translation)

    t_viz_start = time.perf_counter()
    if not args.no_vis:
        rr.set_time_nanos('timestamp', timestamp)
        rr.log('world/trajectory', rr.LineStrips3D(trajectory))
        if USE_SLAM and trajectory_slam:
            rr.log('world/trajectory_slam', rr.LineStrips3D(trajectory_slam))
        rr.log('world/final_landmarks', rr.Points3D(final_landmarks, radii=0.1))
        if USE_SLAM and loop_closure_poses:
            rr.log('world/loop_closure_poses', rr.Points3D(
                loop_closure_poses, radii=1.2, colors=[[255, 0, 0]]
            ))
        rr.log('world/car', rr.Transform3D(
            translation=current_pose.translation,
            quaternion=current_pose.rotation
        ))
        rr.log('world/car/body', rr.Boxes3D(centers=[0, 1.65 / 2, 0], sizes=[[1.6, 1.65, 2.71]]))
        rr.log('world/car/landmarks_center', rr.Points3D(
            landmark_xyz, radii=0.25, colors=landmarks_colors
        ))
        rr.log('world/car/landmarks_lines', rr.Arrows3D(
            vectors=landmark_xyz, radii=0.05, colors=landmarks_colors
        ))
        rr.log('world/car/cam0', rr.Transform3D(
            translation=tf_zedx_left_to_base_link[:3, 3],
            mat3x3=tf_zedx_left_to_base_link[:3, :3]
        ))
        rr.log('world/car/cam0', rr.Pinhole(
            image_plane_distance=1.68,
            focal_length=[left_intrinsics["k"][0], left_intrinsics["k"][4]],
            principal_point=[left_intrinsics["k"][2], left_intrinsics["k"][5]],
            width=size[0],
            height=size[1]
        ))
        rr.log('world/car/cam0/image', rr.Image(images[0]))
        rr.log('world/car/cam0/observations', rr.Points2D(
            observations_uv, radii=5, colors=observations_colors
        ))
        
        if gravity is not None:
            rr.log('world/car/gravity', rr.Arrows3D(vectors=[gravity], colors=[[255, 0, 0]], radii=0.05))
    t_viz_end = time.perf_counter()

    current_camera_frame_time = time.perf_counter()
    loop_time_ms = (current_camera_frame_time - last_camera_frame_time) * 1000.0
    last_camera_frame_time = current_camera_frame_time

    timing_logs[-1]['full_frame_time_ms'] = (t_viz_end - t_frame_start) * 1000.0
    timing_logs[-1]['viz_time_ms'] = (t_viz_end - t_viz_start) * 1000.0
    timing_logs[-1]['loop_time_ms'] = loop_time_ms
    timing_logs[-1]['sleep_time_ms'] = total_sleep_time_since_last_cam * 1000.0
    
    total_sleep_time_since_last_cam = 0.0

    if initial_map_size is None:
        if args.localize:
            if localization_complete.is_set():
                initial_map_size = num_final_landmarks
                try:
                    import lmdb
                    env = lmdb.open(map_path, readonly=True, lock=False, max_dbs=10)
                    with env.begin() as txn:
                        sub_db = env.open_db(b"landmarks", txn=txn)
                        total_db_landmarks = txn.stat(sub_db)['entries']
                except Exception as e:
                    print(f"Failed to read LMDB: {e}")
                    total_db_landmarks = 0

                print(f"\n[Verification] Localization succeeded!")
                print(f"  -> Local active map window: {initial_map_size} landmarks")
                print(f"  -> Total loaded map database: {total_db_landmarks} landmarks")
            elif frame_idx == 0:
                # Set a fallback in case localization never completes, so it isn't None at the end
                initial_map_size = num_final_landmarks
        else:
            initial_map_size = num_final_landmarks
            print(f"\n[Verification] Mapping started. Initial landmarks: {initial_map_size}")

    if frame_idx % 50 == 0 or frame_idx == len(timestamps) - 1:
        recent_logs = timing_logs[-50:] if len(timing_logs) >= 50 else timing_logs
        avg_odom = sum(l['odom_time_ms'] for l in recent_logs) / len(recent_logs)
        avg_slam = sum(l['slam_time_ms'] for l in recent_logs) / len(recent_logs)
        avg_full = sum(l['full_frame_time_ms'] for l in recent_logs) / len(recent_logs)
        avg_viz = sum(l['viz_time_ms'] for l in recent_logs) / len(recent_logs)
        avg_loop = sum(l['loop_time_ms'] for l in recent_logs) / len(recent_logs)
        avg_sleep = sum(l['sleep_time_ms'] for l in recent_logs) / len(recent_logs)
        loc_status = ""
        if args.localize:
            loc_status = " | LOC: OK" if localization_complete.is_set() else " | LOC: pending"
        print(f"[Progress] Frame {frame_idx}/{len(timestamps)} | Odom: {avg_odom:.1f}ms | SLAM: {avg_slam:.1f}ms | Full(w/ viz): {avg_full:.1f}ms | Loop(Total): {avg_loop:.1f}ms | Obs: {len(observations)} | Map LMs: {num_final_landmarks}{loc_status}")
        print(f"  -> Breakdown: Img: {img_time_ms:.1f}ms, Fetch: {fetch_time_ms:.1f}ms, Transform: {transform_time_ms:.1f}ms, Viz: {avg_viz:.1f}ms, Sleep(total): {avg_sleep:.1f}ms")

    frame_idx += 1

print(f"\nNumber of loop closure poses: {len(loop_closure_poses)}")

final_map_size = num_final_landmarks if 'num_final_landmarks' in locals() else 0
print(f"[Verification] Final Map Size: {final_map_size} landmarks.")
if initial_map_size is not None:
    added_landmarks = final_map_size - initial_map_size
    print(f"[Verification] Landmarks added during sequence: {added_landmarks}")
    if args.localize:
        if added_landmarks < 500:
            print("[Verification] STATUS: SUCCESS (System correctly operated in localization mode)")
        else:
            print("[Verification] STATUS: WARNING (System added too many landmarks, may have fallen back to mapping)")

if args.localize:
    print("\n[Verification] Saving map to temporary directory to verify final map size in RAM...")
    temp_map_path = map_path + "_temp"
    os.makedirs(temp_map_path, exist_ok=True)
    map_saved = False
    tracker.save_map(temp_map_path, save_callback)

    start_time = time.time()
    while not map_saved and (time.time() - start_time) < max_wait_time:
        time.sleep(0.1)

    if map_saved:
        try:
            import lmdb
            env = lmdb.open(temp_map_path, readonly=True, lock=False, max_dbs=10)
            with env.begin() as txn:
                sub_db = env.open_db(b"landmarks", txn=txn)
                final_db_landmarks = txn.stat(sub_db)['entries']

            print(f"[Verification] Initial map database had: {total_db_landmarks} landmarks.")
            print(f"[Verification] Final map database has: {final_db_landmarks} landmarks.")
            diff = final_db_landmarks - total_db_landmarks
            print(f"[Verification] Difference: {diff} landmarks.")
        except Exception as e:
            print(f"Failed to read temp map: {e}")

        import shutil
        shutil.rmtree(temp_map_path, ignore_errors=True)
    else:
        print("[Verification] Failed to save temporary map.")

# Modify output path if localizing
if args.output_filepath and guess_pose is not None:
    seq_name = os.path.basename(os.path.normpath(args.sequence_dir))
    args.output_filepath = os.path.join(args.output_filepath, "loc_" + seq_name)

# Save timing logs
timing_dir = args.output_filepath if args.output_filepath else "."
os.makedirs(timing_dir, exist_ok=True)
timing_csv_file = os.path.join(timing_dir, "timing_logs.csv")
with open(timing_csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['frame', 'timestamp', 'odom_time_ms', 'slam_time_ms', 'total_time_ms', 'full_frame_time_ms', 'viz_time_ms', 'loop_time_ms', 'sleep_time_ms'])
    writer.writeheader()
    writer.writerows(timing_logs)
print(f"[Save] Saved timing logs to {timing_csv_file}")

if args.output_filepath:
    os.makedirs(args.output_filepath, exist_ok=True)
    
    if USE_SLAM:
        out_traj_file = os.path.join(args.output_filepath, 'trajectory_tum.txt')
        print(f"[Save] Saving trajectory to {out_traj_file} (length {len(trajectory_tum)})")
        with open(out_traj_file, 'w') as f:
            for item in trajectory_tum:
                f.write(f"{item[0]:.6f} {item[1]:.9f} {item[2]:.9f} {item[3]:.9f} {item[4]:.9f} {item[5]:.9f} {item[6]:.9f} {item[7]:.9f}\n")

    out_odom_traj_file = os.path.join(args.output_filepath, 'trajectory_odom_tum.txt')
    print(f"[Save] Saving Odom trajectory to {out_odom_traj_file} (length {len(trajectory_odom_tum)})")
    with open(out_odom_traj_file, 'w') as f:
        for item in trajectory_odom_tum:
            f.write(f"{item[0]:.6f} {item[1]:.9f} {item[2]:.9f} {item[3]:.9f} {item[4]:.9f} {item[5]:.9f} {item[6]:.9f} {item[7]:.9f}\n")

    if USE_SLAM:
        lc_file = os.path.join(args.output_filepath, "loop_closures.json")
        with open(lc_file, "w") as f:
            json.dump(loop_closures_log, f, indent=4)
        print(f"[Save] Saved {len(loop_closures_log)} loop closures to {lc_file}")

    if USE_SLAM and guess_pose is None:
        os.makedirs(map_path, exist_ok=True)
        tracker.save_map(map_path, save_callback)

        start_time = time.time()
        while not map_saved and (time.time() - start_time) < max_wait_time:
            time.sleep(0.1)

        if map_saved:
            print(f"[Save] Map saved successfully to {map_path}")
        else:
            print("[Warning] Map saving may not have completed")

executor.shutdown(wait=False)

print("Cleaning up resources...")
try:
    del trajectory
    del trajectory_slam
    del trajectory_tum
    del trajectory_odom_tum
    del loop_closure_poses
    del tracker
    del cameras
    del cfg
    if USE_SLAM:
        del s_cfg
except Exception as e:
    print(f"Warning during cleanup: {e}")

print("Script completed")

if args.localize and not localization_complete.is_set():
    import sys
    print("Error: Failed to localize across the entire sequence.")
    sys.exit(1)
