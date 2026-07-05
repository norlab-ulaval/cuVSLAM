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

from fomo_sdk.tf.utils import FoMoTFTree
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description="Track FOMO dataset sequence with SLAM")
parser.add_argument("--sequence_dir", type=str, required=True, help="Path to the sequence directory")
parser.add_argument("--start", type=int, default=0, help="Timestamp of the image to start the export [ms].")
parser.add_argument("--end", type=int, default=np.iinfo(np.int32).max, help="Timestamp of the image to end the export [ms].")
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

tf_tree = FoMoTFTree()


def transform_to_pose(transform_matrix: np.ndarray) -> cuvslam.Pose:
    """Convert a 4x4 transformation matrix to a cuvslam.Pose object."""
    rotation_quat = Rotation.from_matrix(transform_matrix[:3, :3]).as_quat()
    return cuvslam.Pose(rotation=rotation_quat, translation=transform_matrix[:3, 3])


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
                    rrb.Spatial2DView(name="Left Camera", origin='world/car/cam0'),
                    rrb.Spatial2DView(name="Right Camera", origin='world/car/cam1')
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

print(tf_zedx_left_to_base_link)
print()
print(tf_zedx_right_to_base_link)

cameras[0].rig_from_camera = transform_to_pose(tf_zedx_left_to_base_link)
cameras[1].rig_from_camera = transform_to_pose(tf_zedx_right_to_base_link)

left_dir = os.path.join(sequence_path, "zedx_left")
right_dir = os.path.join(sequence_path, "zedx_right")

all_filenames = sorted(os.listdir(left_dir))

start_ns = args.start * 1_000_000
end_ns = args.end * 1_000_000 if args.end != np.iinfo(np.int32).max else float('inf')

filenames = []
for f in all_filenames:
    t_ns = int(os.path.splitext(f)[0]) * 1000
    if start_ns <= t_ns <= end_ns:
        filenames.append(f)

if not filenames:
    print(f"No images found between timestamps {args.start} and {args.end} ms.")
    exit(0)

size = Image.open(os.path.join(left_dir, filenames[0])).size

with open(os.path.join(calib_path, 'zedx_left.json'), 'r') as f:
    left_intrinsics = json.load(f)
with open(os.path.join(calib_path, 'zedx_right.json'), 'r') as f:
    right_intrinsics = json.load(f)

for i, intrinsics in enumerate([left_intrinsics, right_intrinsics]):
    cameras[i].size = size
    cameras[i].focal = [intrinsics["k"][0], intrinsics["k"][4]]
    cameras[i].principal = [intrinsics["k"][2], intrinsics["k"][5]]

rig = cuvslam.Rig()
rig.cameras = cameras

cfg = cuvslam.Tracker.OdometryConfig(
    async_sba=False,
    enable_final_landmarks_export=False,
    rectified_stereo_camera=True,
    odometry_mode=cuvslam.Tracker.OdometryMode.Multicamera
)

tracker = cuvslam.Tracker(rig, cfg)

timestamps = [int(os.path.splitext(f)[0]) * 1000 for f in filenames]

# Prepare metadata
frames_metadata = []
for f_name in filenames:
    t_ns = int(os.path.splitext(f_name)[0]) * 1000
    frames_metadata.append({
        'type': 'camera',
        'timestamp': t_ns,
        'filename': f_name,
        'images_paths': [
            os.path.join(left_dir, f_name),
            os.path.join(right_dir, f_name)
        ]
    })

frames_metadata.sort(key=lambda x: x['timestamp'])

executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

def load_images(image_paths):
    return [asarray(Image.open(p).convert('L')) for p in image_paths]

# Bounded queue to preload/preprocess input images
image_queue = queue.Queue(maxsize=100)

def image_producer():
    try:
        for meta in frames_metadata:
            future = executor.submit(load_images, meta['images_paths'])
            image_queue.put((meta['timestamp'], future))
    except Exception:
        pass

producer_thread = threading.Thread(target=image_producer, daemon=True)
producer_thread.start()

frame_idx = 0
timing_logs = []

if args.output_filepath:
    os.makedirs(os.path.join(args.output_filepath, "zedx_left"), exist_ok=True)
    os.makedirs(os.path.join(args.output_filepath, "zedx_right"), exist_ok=True)

slam_initial_pose = cuvslam.Pose(translation=[0, 0, 0], rotation=[0, 0, 0, 1])

# Main tracking loop
for metadata in frames_metadata:
    timestamp = metadata['timestamp']

    # Camera frame
    t_frame_start = time.perf_counter()
    q_timestamp, image_future = image_queue.get()
    assert q_timestamp == timestamp, f"Timestamp mismatch: queue {q_timestamp} vs frame {timestamp}"
    images = image_future.result()
    t_img_end = time.perf_counter()

    odometry_pose_estimate = tracker.odom.track(timestamp, images, None, None, None)
    t_odom_end = time.perf_counter()
    
    img_time_ms = (t_img_end - t_frame_start) * 1000.0
    odom_time_ms = (t_odom_end - t_img_end) * 1000.0

    timing_logs.append({
        'frame': frame_idx,
        'timestamp': f"{(timestamp / 1_000_000_000.0):.6f}",
        'odom_time_ms': odom_time_ms,
        'full_frame_time_ms': 0.0
    })

    current_pose = slam_initial_pose
    if odometry_pose_estimate.world_from_rig is None:
        print(f"Warning: Failed to track frame {frame_idx}")
    else:
        odom_pose = odometry_pose_estimate.world_from_rig.pose
        current_pose = combine_poses(slam_initial_pose, odom_pose)

    t_fetch_start = time.perf_counter()
    observations_left = tracker.get_last_observations(0)
    observations_right = tracker.get_last_observations(1)
    fetch_time_ms = (time.perf_counter() - t_fetch_start) * 1000.0

    if args.output_filepath:
        for cam_idx, obs in enumerate([observations_left, observations_right]):
            folder = "zedx_left" if cam_idx == 0 else "zedx_right"
            img = np.zeros((size[1], size[0]), dtype=np.uint8)
            for o in obs:
                u, v = int(o.u), int(o.v)
                if 0 <= v < size[1] and 0 <= u < size[0]:
                    img[v, u] = 255
            
            out_path = os.path.join(args.output_filepath, folder, metadata['filename'])
            Image.fromarray(img).save(out_path)

    if not args.no_vis:
        rr.set_time_nanos('timestamp', timestamp)
        
        if odometry_pose_estimate.world_from_rig is not None:
            rr.log('world/car', rr.Transform3D(
                translation=current_pose.translation,
                quaternion=current_pose.rotation
            ))
        rr.log('world/car/body', rr.Boxes3D(centers=[0, 1.65 / 2, 0], sizes=[[1.6, 1.65, 2.71]]))
        
        # Left camera
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
        rr.log('world/car/cam0/image', rr.Image(images[0]).compress(jpeg_quality=80))
        if observations_left:
            obs_uv_left = [[o.u, o.v] for o in observations_left]
            obs_colors_left = [color_from_id(o.id) for o in observations_left]
            rr.log('world/car/cam0/observations', rr.Points2D(
                obs_uv_left, radii=5, colors=obs_colors_left
            ))

        # Right camera
        rr.log('world/car/cam1', rr.Transform3D(
            translation=tf_zedx_right_to_base_link[:3, 3],
            mat3x3=tf_zedx_right_to_base_link[:3, :3]
        ))
        rr.log('world/car/cam1', rr.Pinhole(
            image_plane_distance=1.68,
            focal_length=[right_intrinsics["k"][0], right_intrinsics["k"][4]],
            principal_point=[right_intrinsics["k"][2], right_intrinsics["k"][5]],
            width=size[0],
            height=size[1]
        ))
        rr.log('world/car/cam1/image', rr.Image(images[1]).compress(jpeg_quality=80))
        if observations_right:
            obs_uv_right = [[o.u, o.v] for o in observations_right]
            obs_colors_right = [color_from_id(o.id) for o in observations_right]
            rr.log('world/car/cam1/observations', rr.Points2D(
                obs_uv_right, radii=5, colors=obs_colors_right
            ))

    timing_logs[-1]['full_frame_time_ms'] = (time.perf_counter() - t_frame_start) * 1000.0

    if frame_idx % 50 == 0 or frame_idx == len(timestamps) - 1:
        recent_logs = timing_logs[-50:] if len(timing_logs) >= 50 else timing_logs
        avg_odom = sum(l['odom_time_ms'] for l in recent_logs) / len(recent_logs)
        avg_full = sum(l['full_frame_time_ms'] for l in recent_logs) / len(recent_logs)
        print(f"[Progress] Frame {frame_idx}/{len(timestamps)} | Odom: {avg_odom:.1f}ms | Full: {avg_full:.1f}ms | Obs L: {len(observations_left)} | Obs R: {len(observations_right)}")
        print(f"  -> Breakdown of current frame: Img: {img_time_ms:.1f}ms, Fetch: {fetch_time_ms:.1f}ms")

    frame_idx += 1

# Save timing logs
timing_dir = args.output_filepath if args.output_filepath else "."
os.makedirs(timing_dir, exist_ok=True)
timing_csv_file = os.path.join(timing_dir, "timing_logs.csv")
with open(timing_csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['frame', 'timestamp', 'odom_time_ms', 'full_frame_time_ms'])
    writer.writeheader()
    for log in timing_logs:
        writer.writerow({k: log[k] for k in writer.fieldnames})
print(f"[Save] Saved timing logs to {timing_csv_file}")

executor.shutdown(wait=False)

print("Cleaning up resources...")
try:
    del tracker
    del cameras
    del cfg
except Exception as e:
    print(f"Warning during cleanup: {e}")

print("Script completed")
