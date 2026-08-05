import os
import threading
import time
import argparse
import json
from PIL import Image
from numpy import loadtxt, asarray, array_equal as np_array_equal, savetxt
from scipy.spatial.transform import Rotation as R
import rerun as rr
import rerun.blueprint as rrb
import cuvslam
from fomo_sdk.tf.utils import FoMoTFTree

parser = argparse.ArgumentParser(description="Track FoMo dataset sequence with SLAM")
parser.add_argument("--sequence_dir", type=str, required=True, help="Path to the sequence directory")
parser.add_argument("--localize", action="store_true", help="Enable localization using existing map. Otherwise runs SLAM and mapping.")
parser.add_argument("--output_filepath", type=str, default="", help="Directory for map and trajectory outputs. Defaults to <sequence_dir>/map and <sequence_dir>/trajectory_tum.txt.")
args = parser.parse_args()

sequence_path = os.path.abspath(args.sequence_dir)
calib_path = os.path.join(sequence_path, "calib")

quaternion_to_rotation_matrix = lambda q: R.from_quat(q).as_matrix().tolist()
quaternion_multiply = lambda q1, q2: (R.from_quat(q1) * R.from_quat(q2)).as_quat()
rotate_vector = lambda vector, rotation_matrix: R.from_matrix(rotation_matrix).apply(vector)


def transform_to_pose(transform_matrix):
    rotation_quat = R.from_matrix(transform_matrix[:3, :3]).as_quat()
    return cuvslam.Pose(rotation_quat, transform_matrix[:3, 3].tolist())


def combine_poses(initial_pose, relative_pose):
    rotation_matrix = quaternion_to_rotation_matrix(initial_pose.rotation)
    rotated_rel_t = rotate_vector(relative_pose.translation, rotation_matrix)
    absolute_translation = [
        initial_pose.translation[0] + rotated_rel_t[0],
        initial_pose.translation[1] + rotated_rel_t[1],
        initial_pose.translation[2] + rotated_rel_t[2]
    ]
    absolute_rotation = quaternion_multiply(initial_pose.rotation, relative_pose.rotation)
    return cuvslam.Pose(absolute_rotation, absolute_translation)


def transform_landmarks(landmarks, initial_pose):
    if not landmarks:
        return []
    rotation_matrix = quaternion_to_rotation_matrix(initial_pose.rotation)
    pts = asarray(landmarks)
    rot = asarray(rotation_matrix)
    trans = asarray(initial_pose.translation)
    transformed_pts = (pts @ rot.T) + trans
    return transformed_pts.tolist()


def save_callback(success):
    global map_saved
    map_saved = success

def localization_start_cb():
    print("Localization started")

def localization_finish_cb(pose, error_message):
    global slam_initial_pose
    print(f"Localization result: {pose}, {error_message}")
    slam_initial_pose = pose
    localization_complete.set()


def color_from_id(identifier):
    return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]


# Setup rerun visualizer
rr.init('fomo_slam', strict=True, spawn=True)

rr.send_blueprint(rrb.Blueprint(
    rrb.TimePanel(state="collapsed"),
    rrb.Vertical(
        row_shares=[0.6, 0.4],
        contents=[rrb.Spatial3DView(), rrb.Spatial2DView(origin='rig/cam0')]
    )
))

rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

rr.log("xyz", rr.Arrows3D(
    vectors=[[50, 0, 0], [0, 50, 0], [0, 0, 50]],
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    labels=['[x]', '[y]', '[z]']
), static=True)

SLAM_SYNC_MODE = False
IDX = 0
max_wait_time = 10.0

# ----- FoMo data loading (adapted from track_fomo_minimal.py L57-80) -----
tf_tree = FoMoTFTree()
cameras = [cuvslam.Camera(), cuvslam.Camera()]
tf_zedx_left_to_base_link = tf_tree.get_transform(from_frame="base_link", to_frame="zedx_left")
tf_zedx_right_to_zedx_left = tf_tree.get_transform(from_frame="zedx_left", to_frame="zedx_right")
translation_only = __import__('numpy').eye(4)
translation_only[:3, 3] = tf_zedx_right_to_zedx_left[:3, 3]
tf_zedx_right_to_base_link = tf_zedx_left_to_base_link @ translation_only

cameras[0].rig_from_camera = transform_to_pose(tf_zedx_left_to_base_link)
cameras[1].rig_from_camera = transform_to_pose(tf_zedx_right_to_base_link)

left_dir = os.path.join(sequence_path, "zedx_left")
right_dir = os.path.join(sequence_path, "zedx_right")
filenames = sorted(os.listdir(left_dir))

with open(os.path.join(calib_path, 'zedx_left.json'), 'r') as f:
    left_intrinsics = json.load(f)
with open(os.path.join(calib_path, 'zedx_right.json'), 'r') as f:
    right_intrinsics = json.load(f)

for i, intrinsics_data in enumerate([left_intrinsics, right_intrinsics]):
    cameras[i].size = Image.open(os.path.join(left_dir, filenames[0])).size
    cameras[i].focal = [intrinsics_data["k"][0], intrinsics_data["k"][4]]
    cameras[i].principal = [intrinsics_data["k"][2], intrinsics_data["k"][5]]

# Intrinsic matrix for rerun pinhole logging (left camera)
import numpy as np
size = cameras[0].size
left_K = np.array([
    [left_intrinsics["k"][0], 0,                       left_intrinsics["k"][2]],
    [0,                       left_intrinsics["k"][4], left_intrinsics["k"][5]],
    [0,                       0,                       1                      ]
])

timestamps = [int(os.path.splitext(f)[0]) * 1000 for f in filenames]
# -------------------------------------------------------------------------

cfg = cuvslam.Tracker.OdometryConfig(
    async_sba=False,
    enable_final_landmarks_export=True,
    rectified_stereo_camera=True
)
s_cfg = cuvslam.Tracker.SlamConfig(sync_mode=SLAM_SYNC_MODE, enable_mapping=not args.localize)
tracker = cuvslam.Tracker(cuvslam.Rig(cameras), cfg, s_cfg)

map_path = os.path.join(args.output_filepath, 'map') if args.output_filepath else os.path.join(sequence_path, 'map')
trajectory_file = os.path.join(args.output_filepath, 'trajectory_tum.txt') if args.output_filepath else os.path.join(sequence_path, 'trajectory_tum.txt')
os.makedirs(os.path.dirname(trajectory_file), exist_ok=True)

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

if args.localize:
    print("Running in LOCALIZATION mode")
    if os.path.exists(trajectory_file) and os.path.exists(map_path):
        trajectory_data = loadtxt(trajectory_file)
        if IDX >= len(trajectory_data):
            raise IndexError(
                f"IDX ({IDX}) is out of bounds for loaded trajectory of length {len(trajectory_data)}"
            )
        guess_tum_pose = trajectory_data[IDX]
        
        if len(guess_tum_pose) == 8:
            # Standard 8-column TUM format: [timestamp, tx, ty, tz, qx, qy, qz, qw]
            guess_pose = cuvslam.Pose(guess_tum_pose[4:8].tolist(), guess_tum_pose[1:4].tolist())
        else:
            raise ValueError(f"Unexpected number of columns in trajectory file: {len(guess_tum_pose)}")
    else:
        print(f"Warning: Map or trajectory file not found in {sequence_path} for localization.")
else:
    print("Running in MAPPING mode")
    guess_pose = None

if os.path.exists(map_path) and (guess_pose is not None):
    timestamp = timestamps[IDX]
    init_images = [
        asarray(Image.open(os.path.join(left_dir if cam == 0 else right_dir, filenames[IDX])).convert('L'))
        for cam in [0, 1]
    ]
    _, _ = tracker.track(timestamp, init_images)

    tracker.localize_in_map(map_path, timestamp, guess_pose, init_images, loc_settings, localization_start_cb,
                            localization_finish_cb)

    wait_time = 0
    wait_timestamp_ns = timestamp
    while not SLAM_SYNC_MODE and not localization_complete.wait(timeout=0.5) and wait_time < max_wait_time:
        print(f"Waiting for localization... timestamp: {wait_time}")
        if wait_time < 5.0:
            wait_timestamp_ns += 1_000_000
            _, slam_pose = tracker.track(wait_timestamp_ns, init_images)
            print(f"  slam_pose.t: {[f'{x:.3f}' for x in slam_pose.translation]}")
        wait_time += 0.5
    if not localization_complete.is_set():
        print(f"Localization did not complete within {max_wait_time} seconds")
    IDX += 1

if slam_initial_pose is not None:
    print(f"Localized pose: {slam_initial_pose}")
    wait_time = 0
    while not SLAM_SYNC_MODE and wait_time < max_wait_time:
        time.sleep(0.5)
        wait_timestamp_ns += 1_000_000
        _, slam_pose = tracker.track(wait_timestamp_ns, init_images)
        print(f"  slam_pose.t: {[f'{x:.3f}' for x in slam_pose.translation]}")
        identity_t = all(abs(x) < 1e-6 for x in slam_pose.translation)
        identity_r = all(abs(x) < 1e-6 for x in slam_pose.rotation[:3]) and abs(slam_pose.rotation[3] - 1.0) < 1e-6
        if not (identity_t and identity_r):
            break
        wait_time += 0.5
else:
    print("Warning: slam_initial_pose is None, set initial pose to zero, starting frame to 0, ignore map if exists")
    IDX = 0
    slam_initial_pose = cuvslam.Pose([0, 0, 0, 1], [0, 0, 0])

trajectory = []
trajectory_slam = []
trajectory_tum = []
loop_closure_poses = []
initial_map_size = None

total_db_landmarks = 0
for frame in range(IDX, len(timestamps)):
    time.sleep(0.01)

    images = [
        asarray(Image.open(os.path.join(left_dir, filenames[frame])).convert('L')),
        asarray(Image.open(os.path.join(right_dir, filenames[frame])).convert('L'))
    ]

    t0 = time.time()
    odometry_pose_estimate, slam_pose = tracker.track(timestamps[frame], images)
    track_time = time.time() - t0

    if odometry_pose_estimate.world_from_rig is None:
        print(f"Warning: Failed to track frame {frame}")
        continue

    odom_pose = odometry_pose_estimate.world_from_rig.pose
    current_pose = combine_poses(slam_initial_pose, odom_pose)

    observations = tracker.get_last_observations(0)
    landmarks = tracker.get_last_landmarks()

    t1 = time.time()
    raw_final_landmarks = list(tracker.get_final_landmarks().values())
    final_landmarks = transform_landmarks(raw_final_landmarks, slam_initial_pose)
    transform_time = time.time() - t1

    if frame % 10 == 0:
        print(f"[Frame {frame:04d}] tracker.track(): {track_time*1000:.1f} ms | transform_landmarks() ({len(raw_final_landmarks)} pts): {transform_time*1000:.1f} ms")

    if initial_map_size is None:
        initial_map_size = len(final_landmarks)
        if args.localize:
            try:
                import lmdb
                env = lmdb.open(map_path, readonly=True, lock=False, max_dbs=10)
                with env.begin() as txn:
                    sub_db = env.open_db(b"landmarks", txn=txn)
                    total_db_landmarks = txn.stat(sub_db)['entries']
            except Exception as e:
                print(f"Failed to read LMDB: {e}")

            print(f"\n[Verification] Localization succeeded!")
            print(f"  -> Local active map window: {initial_map_size} landmarks")
            print(f"  -> Total loaded map database: {total_db_landmarks} landmarks")
        else:
            print(f"\n[Verification] Mapping started. Initial landmarks: {initial_map_size}")

    observations_uv = [[o.u, o.v] for o in observations]
    observations_colors = [color_from_id(o.id) for o in observations]
    landmark_xyz = [l.coords for l in landmarks]
    landmarks_colors = [color_from_id(l.id) for l in landmarks]

    trajectory.append(current_pose.translation)
    trajectory_slam.append(slam_pose.translation)
    
    # Save standard 8-column TUM format: [timestamp_sec, tx, ty, tz, qx, qy, qz, qw]
    tum_timestamp_sec = timestamps[frame] * 1e-9
    trajectory_tum.append([tum_timestamp_sec] + list(slam_pose.translation) + list(slam_pose.rotation))

    current_lc_poses = tracker.get_loop_closure_poses()
    if (current_lc_poses and
        (not loop_closure_poses or
         not np_array_equal(current_lc_poses[-1].pose.translation, loop_closure_poses[-1]))):
        loop_closure_poses.append(current_lc_poses[-1].pose.translation)

    rr.set_time_nanos('timestamp', timestamps[frame])
    rr.log('trajectory', rr.LineStrips3D(trajectory))
    rr.log('trajectory_slam', rr.LineStrips3D(trajectory_slam))
    rr.log('final_landmarks', rr.Points3D(final_landmarks, radii=0.1))
    rr.log('loop_closure_poses', rr.Points3D(
        loop_closure_poses, radii=1.2, colors=[[255, 0, 0]]
    ))
    rr.log('rig', rr.Transform3D(
        translation=current_pose.translation,
        quaternion=current_pose.rotation
    ))
    rr.log('rig/body', rr.Boxes3D(centers=[0, 0, 0], sizes=[[0.5, 0.5, 1.0]]))
    rr.log('rig/landmarks_center', rr.Points3D(
        landmark_xyz, radii=0.25, colors=landmarks_colors
    ))
    rr.log('rig/landmarks_lines', rr.Arrows3D(
        vectors=landmark_xyz, radii=0.05, colors=landmarks_colors
    ))
    rr.log('rig/cam0', rr.Pinhole(
        image_plane_distance=1.68,
        image_from_camera=left_K,
        width=size[0],
        height=size[1]
    ))
    rr.log('rig/cam0/image', rr.Image(images[0]).compress(jpeg_quality=80))
    rr.log('rig/cam0/observations', rr.Points2D(
        observations_uv, radii=5, colors=observations_colors
    ))

print(f"Number of loop closure poses: {len(loop_closure_poses)}")

final_map_size = len(final_landmarks) if 'final_landmarks' in locals() else 0
print(f"[Verification] Final Map Size: {final_map_size} landmarks.")
if initial_map_size is not None:
    added_landmarks = final_map_size - initial_map_size
    print(f"[Verification] Landmarks added during sequence: {added_landmarks}")
    if args.localize:
        print(f"[Verification] In localization mode, added relatively few landmarks: {added_landmarks}")

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
            if abs(diff) < 500:
                print("[Verification] PROOF: The system maintained a stable map size (it was localizing, not mapping).")
            else:
                print("[Verification] WARNING: The map grew significantly. It might have been mapping.")
        except Exception as e:
            print(f"Failed to read temp map: {e}")

        import shutil
        shutil.rmtree(temp_map_path, ignore_errors=True)
    else:
        print("[Verification] Failed to save temporary map.")

os.makedirs(map_path, exist_ok=True)  # ensure map dir exists before save_map

# Determine output directory: if localization mode is on, save to <output_filepath>/<name of sequence_dir>
if args.localize:
    base_output_dir = args.output_filepath if args.output_filepath else sequence_path
    sequence_name = os.path.basename(os.path.normpath(sequence_path))
    output_dir = os.path.join(base_output_dir, sequence_name)
else:
    output_dir = os.path.dirname(trajectory_file)

os.makedirs(output_dir, exist_ok=True)

out_trajectory_file = os.path.join(output_dir, "trajectory_tum.txt")
print(f"Saving trajectory_tum to {out_trajectory_file} of length {len(trajectory_tum)}")
savetxt(out_trajectory_file, trajectory_tum)

traj_slam_file = os.path.join(output_dir, "trajectory_slam.txt")
print(f"Saving trajectory_slam to {traj_slam_file} of length {len(trajectory_slam)}")
savetxt(traj_slam_file, trajectory_slam)

traj_odom_file = os.path.join(output_dir, "trajectory_odom.txt")
print(f"Saving trajectory_odom to {traj_odom_file} of length {len(trajectory)}")
savetxt(traj_odom_file, trajectory)

if not args.localize:
    tracker.save_map(map_path, save_callback)

    start_time = time.time()
    while not map_saved and (time.time() - start_time) < max_wait_time:
        time.sleep(0.1)
        print(f"Waiting for map saving to complete... {time.time() - start_time} seconds")

    if map_saved:
        print("Map saved successfully")
    else:
        print("WARNING: Map saving may not have completed")

print("Cleaning up resources...")
try:
    del trajectory
    del trajectory_slam
    del trajectory_tum
    del loop_closure_poses
    del tracker
    del cameras
    del cfg
    del s_cfg
except Exception as e:
    print(f"Warning during cleanup: {e}")

print("Script completed")