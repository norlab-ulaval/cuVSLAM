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
from numpy import asarray
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb
import cuvslam

# Parse arguments
parser = argparse.ArgumentParser(description="Track FOMO dataset sequence")
parser.add_argument("--sequence", type=str, default="00", choices=["00", "01"], help="Sequence to track")
args = parser.parse_args()

# Set up dataset path
dataset_path = os.path.join(os.path.dirname(__file__), "dataset")
sequence_path = os.path.join(dataset_path, "sequences", args.sequence)

# Generate pseudo-random colour from integer identifier for visualization
def color_from_id(identifier):
    return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]

# Setup rerun visualizer
rr.init('fomo', strict=True, spawn=True)  # launch re-run instance

# Setup rerun views
rr.send_blueprint(rrb.Blueprint(
    rrb.TimePanel(state="collapsed"),
    rrb.Vertical(
        row_shares=[0.6, 0.4],
        contents=[rrb.Spatial3DView(), rrb.Spatial2DView(origin='car/cam0')]
    )
))

# Setup coordinate basis for root, cuvslam uses right-hand system with X-right, Y-down, Z-forward
rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# Draw arrays in origin X-red, Y-green, Z-blue
rr.log("xyz", rr.Arrows3D(
    vectors=[[50, 0, 0], [0, 50, 0], [0, 0, 50]],
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    labels=['[x]', '[y]', '[z]']
), static=True)

# Load FOMO dataset transforms
with open(os.path.join(dataset_path, 'transforms.json'), 'r') as f:
    transforms = json.load(f)

cameras = [cuvslam.Camera(), cuvslam.Camera()]

for t in transforms:
    if t["to"] == "zedx_right" and t["from"] == "zedx_left":
        p = t["position"]
        # For rectified stereo, relative rotation must be identity and translation purely along the X-axis (baseline)
        cameras[1].rig_from_camera.translation = [p["x"], 0.0, 0.0]
        break

left_dir = os.path.join(sequence_path, "zedx_left")
right_dir = os.path.join(sequence_path, "zedx_right")

# Get filenames and sort them to ensure correct temporal order
filenames = sorted(os.listdir(left_dir))

# Get size from the first image
size = Image.open(os.path.join(left_dir, filenames[0])).size

with open(os.path.join(dataset_path, 'zedx_left.json'), 'r') as f:
    left_intrinsics = json.load(f)
with open(os.path.join(dataset_path, 'zedx_right.json'), 'r') as f:
    right_intrinsics = json.load(f)

for i, intrinsics in enumerate([left_intrinsics, right_intrinsics]):
    cameras[i].size = size
    cameras[i].focal = [intrinsics["k"][0], intrinsics["k"][4]]
    cameras[i].principal = [intrinsics["k"][2], intrinsics["k"][5]]

# Initialize the cuvslam tracker
cfg = cuvslam.Tracker.OdometryConfig(
    async_sba=False,
    enable_final_landmarks_export=True,
    rectified_stereo_camera=True
)
tracker = cuvslam.Tracker(cuvslam.Rig(cameras), cfg)

# Extract timestamps (nanoseconds) from filenames (microseconds)
timestamps = [int(os.path.splitext(f)[0]) * 1000 for f in filenames]

# Track each frames in the dataset sequence
trajectory = []
for frame in range(len(timestamps)):
    f_name = filenames[frame]
    # Load grayscale pixels as array for left and right absolute image paths
    images = [
        asarray(Image.open(os.path.join(left_dir, f_name)).convert('L')),
        asarray(Image.open(os.path.join(right_dir, f_name)).convert('L'))
    ]

    # Do tracking
    odom_pose_estimate, _ = tracker.track(timestamps[frame], images)

    if odom_pose_estimate.world_from_rig is None:
        print(f"Warning: Failed to track frame {frame}")
        continue

    # Get current pose and observations for the main camera and gravity in rig frame
    odom_pose = odom_pose_estimate.world_from_rig.pose

    # Get visualization data
    observations = tracker.get_last_observations(0)  # get observation from left camera
    landmarks = tracker.get_last_landmarks()
    final_landmarks = tracker.get_final_landmarks()

    # Prepare visualization data
    observations_uv = [[o.u, o.v] for o in observations]
    observations_colors = [color_from_id(o.id) for o in observations]
    landmark_xyz = [l.coords for l in landmarks]
    landmarks_colors = [color_from_id(l.id) for l in landmarks]
    trajectory.append(odom_pose.translation)

    # Send results to rerun for visualization
    rr.set_time_sequence('frame', frame)
    rr.log('trajectory', rr.LineStrips3D(trajectory))
    rr.log('final_landmarks', rr.Points3D(list(final_landmarks.values()), radii=0.1))
    rr.log('car', rr.Transform3D(
        translation=odom_pose.translation,
        quaternion=odom_pose.rotation
    ))
    rr.log('car/body', rr.Boxes3D(centers=[0, 1.65 / 2, 0], sizes=[[1.6, 1.65, 2.71]]))
    rr.log('car/landmarks_center', rr.Points3D(
        landmark_xyz, radii=0.25, colors=landmarks_colors
    ))
    rr.log('car/landmarks_lines', rr.Arrows3D(
        vectors=landmark_xyz, radii=0.05, colors=landmarks_colors
    ))
    rr.log('car/cam0', rr.Pinhole(
        image_plane_distance=1.68,
        focal_length=[left_intrinsics["k"][0], left_intrinsics["k"][4]],
        principal_point=[left_intrinsics["k"][2], left_intrinsics["k"][5]],
        width=size[0],
        height=size[1]
    ))
    rr.log('car/cam0/image', rr.Image(images[0]).compress(jpeg_quality=80))
    rr.log('car/cam0/observations', rr.Points2D(
        observations_uv, radii=5, colors=observations_colors
    ))
