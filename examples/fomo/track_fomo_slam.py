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

parser = argparse.ArgumentParser(description="Track FOMO dataset sequence with SLAM")
parser.add_argument("--sequence_dir", type=str, required=True, help="Path to the sequence directory")
parser.add_argument("--slam_sync_mode", action="store_true", help="Enable sync slam thread")
parser.add_argument("--idx", type=int, default=700, help="Starting index of the sequence after localization. If negative, don't localize but run SLAM and map.")
parser.add_argument("--max_wait_time", type=float, default=10.0, help="Max wait time in seconds")
parser.add_argument("--output_filepath", type=str, default="", help="Output filepath. If empty, don't save.")
parser.add_argument("--no_vis", action="store_true", help="Disable rerun visualization")
args = parser.parse_args()

# Dataset sequence to track and visualize
sequence_path = os.path.abspath(args.sequence_dir)
calib_path = os.path.join(sequence_path, "calib")

# Lambda to convert quaternion [x, y, z, w] to 3x3 rotation matrix (as list of lists)
quaternion_to_rotation_matrix = lambda q: R.from_quat(q).as_matrix().tolist()

# Lambda to multiply two quaternions [x, y, z, w] * [x, y, z, w]
quaternion_multiply = lambda q1, q2: (R.from_quat(q1) * R.from_quat(q2)).as_quat()

# Lambda to rotate a 3D vector using a 3x3 rotation matrix
rotate_vector = lambda vector, rotation_matrix: R.from_matrix(rotation_matrix).apply(vector)


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
                    rrb.Spatial2DView(origin='world/cam0'),
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

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    rr.log("world/xyz", rr.Arrows3D(
        vectors=[[50, 0, 0], [0, 50, 0], [0, 0, 50]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        labels=['[x]', '[y]', '[z]']
    ), static=True)

SLAM_SYNC_MODE = args.slam_sync_mode
IDX = args.idx
max_wait_time = args.max_wait_time

with open(os.path.join(calib_path, 'transforms.json'), 'r') as f:
    transforms = json.load(f)

cameras = [cuvslam.Camera(), cuvslam.Camera()]

for t in transforms:
    if t["to"] == "zedx_right" and t["from"] == "zedx_left":
        p = t["position"]
        cameras[1].rig_from_camera = cuvslam.Pose(
            translation=[p["x"], 0.0, 0.0],
            rotation=[0.0, 0.0, 0.0, 1.0]
        )
        break

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
imu = cuvslam.ImuCalibration()
for t in transforms:
    if t["to"] == "vectornav" and t["from"] == "zedx_left":
        p = t["position"]
        q = t["orientation"]
        imu.rig_from_imu = cuvslam.Pose(
            translation=[p["x"], p["y"], p["z"]],
            rotation=[q["x"], q["y"], q["z"], q["w"]]
        )

imu.gyroscope_noise_density = 5.55e-5
imu.gyroscope_random_walk = 1.05e-6
imu.accelerometer_noise_density = 1.14e-3
imu.accelerometer_random_walk = 1.6e-5
imu.frequency = 200.0

rig = cuvslam.Rig()
rig.cameras = cameras
rig.imus = [imu]

cfg = cuvslam.Tracker.OdometryConfig(
    async_sba=False,
    enable_final_landmarks_export=True,
    rectified_stereo_camera=True,
    odometry_mode=cuvslam.Tracker.OdometryMode.Inertial
)
s_cfg = cuvslam.Tracker.SlamConfig(sync_mode=SLAM_SYNC_MODE)
tracker = cuvslam.Tracker(rig, cfg, s_cfg)

timestamps = [int(os.path.splitext(f)[0]) * 1000 for f in filenames]

# Check if map folder and trajectory file exist
map_path = os.path.join(sequence_path, 'map')
trajectory_file = os.path.join(sequence_path, 'trajectory_tum.txt')

if not os.path.exists(map_path):
    print(f"Map folder not found at {map_path}")

localization_complete = threading.Event()
slam_initial_pose = None
guess_pose = None
map_saved = False

loc_settings = cuvslam.Tracker.SlamLocalizationSettings(
    horizontal_search_radius=8.,
    vertical_search_radius=2.,
    horizontal_step=0.5,
    vertical_step=0.2,
    angular_step_rads=0.03
    )

if IDX < 0:
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

# Localize in map if applicable
if os.path.exists(map_path) and (guess_pose is not None):
    init_images = None
    for meta in frames_metadata:
        if meta['type'] == 'camera' and meta['timestamp'] == start_timestamp_ns:
            init_images = [
                asarray(Image.open(meta['images_paths'][0]).convert('L')),
                asarray(Image.open(meta['images_paths'][1]).convert('L'))
            ]
            break
            
    if init_images is not None:
        _, _ = tracker.track(start_timestamp_ns, init_images)
        tracker.localize_in_map(map_path, start_timestamp_ns, guess_pose, init_images, loc_settings, localization_start_cb, localization_finish_cb)
        
        wait_time = 0
        while not SLAM_SYNC_MODE and not localization_complete.wait(timeout=0.5) and wait_time < max_wait_time:
            print(f"Waiting for localization... {wait_time}s")
            wait_time += 0.5
            
        if not localization_complete.is_set():
            print(f"Localization did not complete within {max_wait_time} seconds")

# Determine fast-forward logic
skip_until_ns = -1
if slam_initial_pose is not None and guess_pose is not None:
    print(f"Localized pose: {slam_initial_pose}")
    skip_until_ns = start_timestamp_ns
else:
    print("Warning: slam_initial_pose is None, set initial pose to zero, starting frame to 0, ignore map if exists")
    slam_initial_pose = cuvslam.Pose(translation=[0, 0, 0], rotation=[0, 0, 0, 1])

trajectory = []
trajectory_slam = []
trajectory_tum = []
trajectory_odom_tum = []
loop_closure_poses = []
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

# Main tracking loop
for metadata in frames_metadata:
    timestamp = metadata['timestamp']

    if timestamp <= skip_until_ns:
        if metadata['type'] == 'camera':
            image_queue.get()
            frame_idx += 1
        continue

    if metadata['type'] == 'imu':
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
    if tracker.slam and odometry_pose_estimate.world_from_rig:
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
        'full_frame_time_ms': 0.0
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
    
    if needs_viz or needs_print:
        raw_final_landmarks = list(tracker.get_final_landmarks().values())
        t_fetch_end = time.perf_counter()
        final_landmarks = transform_landmarks(raw_final_landmarks, slam_initial_pose)
        t_transform_end = time.perf_counter()
        fetch_time_ms = (t_fetch_end - t_fetch_start) * 1000.0
        transform_time_ms = (t_transform_end - t_fetch_end) * 1000.0
    else:
        final_landmarks = []
        fetch_time_ms = (time.perf_counter() - t_fetch_start) * 1000.0
        transform_time_ms = 0.0
        
    gravity = tracker.get_last_gravity()

    if needs_viz:
        observations_uv = [[o.u, o.v] for o in observations]
        observations_colors = [color_from_id(o.id) for o in observations]
        landmark_xyz = [l.coords for l in landmarks]
        landmarks_colors = [color_from_id(l.id) for l in landmarks]

    trajectory.append(current_pose.translation)
    trajectory_slam.append(slam_pose.translation)
    trajectory_tum.append([timestamp / 1_000_000_000.0] + list(slam_pose.translation) + list(slam_pose.rotation))
    trajectory_odom_tum.append([timestamp / 1_000_000_000.0] + list(current_pose.translation) + list(current_pose.rotation))

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

    if not args.no_vis:
        rr.set_time_nanos('timestamp', timestamp)
        rr.log('world/trajectory', rr.LineStrips3D(trajectory))
        rr.log('world/trajectory_slam', rr.LineStrips3D(trajectory_slam))
        rr.log('world/final_landmarks', rr.Points3D(final_landmarks, radii=0.1))
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
        rr.log('world/cam0', rr.Pinhole(
            image_plane_distance=1.68,
            focal_length=[left_intrinsics["k"][0], left_intrinsics["k"][4]],
            principal_point=[left_intrinsics["k"][2], left_intrinsics["k"][5]],
            width=size[0],
            height=size[1]
        ))
        rr.log('world/cam0/image', rr.Image(images[0]).compress(jpeg_quality=80))
        rr.log('world/cam0/observations', rr.Points2D(
            observations_uv, radii=5, colors=observations_colors
        ))
        
        if gravity is not None:
            rr.log('world/car/gravity', rr.Arrows3D(vectors=[gravity], colors=[[255, 0, 0]], radii=0.05))

    timing_logs[-1]['full_frame_time_ms'] = (time.perf_counter() - t_frame_start) * 1000.0

    if frame_idx % 50 == 0 or frame_idx == len(timestamps) - 1:
        recent_logs = timing_logs[-50:] if len(timing_logs) >= 50 else timing_logs
        avg_odom = sum(l['odom_time_ms'] for l in recent_logs) / len(recent_logs)
        avg_slam = sum(l['slam_time_ms'] for l in recent_logs) / len(recent_logs)
        avg_full = sum(l['full_frame_time_ms'] for l in recent_logs) / len(recent_logs)
        print(f"[Progress] Frame {frame_idx}/{len(timestamps)} | Odom: {avg_odom:.1f}ms | SLAM: {avg_slam:.1f}ms | Full: {avg_full:.1f}ms | Obs: {len(observations)} | Map LMs: {len(final_landmarks)}")
        print(f"  -> Breakdown of current frame: Img: {img_time_ms:.1f}ms, Fetch: {fetch_time_ms:.1f}ms, Transform: {transform_time_ms:.1f}ms")

    frame_idx += 1

print(f"Number of loop closure poses: {len(loop_closure_poses)}")

# Save timing logs
timing_dir = args.output_filepath if args.output_filepath else "."
os.makedirs(timing_dir, exist_ok=True)
timing_csv_file = os.path.join(timing_dir, "timing_logs.csv")
with open(timing_csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['frame', 'timestamp', 'odom_time_ms', 'slam_time_ms', 'total_time_ms', 'full_frame_time_ms'])
    writer.writeheader()
    writer.writerows(timing_logs)
print(f"[Save] Saved timing logs to {timing_csv_file}")

if args.output_filepath:
    os.makedirs(args.output_filepath, exist_ok=True)
    
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

    lc_file = os.path.join(args.output_filepath, "loop_closures.json")
    with open(lc_file, "w") as f:
        json.dump(loop_closures_log, f, indent=4)
    print(f"[Save] Saved {len(loop_closures_log)} loop closures to {lc_file}")

    if guess_pose is None:
        temp_map_dir = os.path.join(args.output_filepath, "map_temp")
        os.makedirs(temp_map_dir, exist_ok=True)
        tracker.save_map(temp_map_dir, save_callback)

        start_time = time.time()
        while not map_saved and (time.time() - start_time) < max_wait_time:
            time.sleep(0.1)

        if map_saved:
            print("[Save] Map saved successfully")
            temp_data_file = os.path.join(temp_map_dir, "data.mdb")
            target_map_file = os.path.join(args.output_filepath, "map.mdb")
            if os.path.exists(temp_data_file):
                import shutil
                try:
                    shutil.move(temp_data_file, target_map_file)
                    print(f"[Save] Renamed map database to {target_map_file}")
                except Exception as e:
                    print(f"[Warning] Failed to rename map: {e}")
            try:
                for item in os.listdir(temp_map_dir):
                    os.remove(os.path.join(temp_map_dir, item))
                os.rmdir(temp_map_dir)
            except Exception as e:
                print(f"[Warning] Failed to clean up temp map directory: {e}")
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
    del s_cfg
except Exception as e:
    print(f"Warning during cleanup: {e}")

print("Script completed")
