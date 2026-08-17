# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import os
import argparse
import numpy as np
import cv2
import rerun as rr
import rerun.blueprint as rrb
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import yaml
import concurrent.futures
import queue
import threading
import glob
from tqdm import tqdm

import cuvslam

def color_from_id(identifier):
    """Generate pseudo-random color from integer identifier for visualization."""
    return [
        (identifier * 17) % 256,
        (identifier * 31) % 256,
        (identifier * 47) % 256
    ]

def preprocess_bag(bag_path, base_dataset_dir):
    print(f"Preprocessing {bag_path} into {base_dataset_dir} (front and rear)...")
    front_dir = os.path.join(base_dataset_dir, "front")
    rear_dir = os.path.join(base_dataset_dir, "rear")
    os.makedirs(front_dir, exist_ok=True)
    os.makedirs(rear_dir, exist_ok=True)
    
    topics = {
        "/wide_angle_camera_front/camera_info": (front_dir, "info"),
        "/wide_angle_camera_front/image_color/compressed": (front_dir, "image"),
        "/wide_angle_camera_rear/camera_info": (rear_dir, "info"),
        "/wide_angle_camera_rear/image_color/compressed": (rear_dir, "image")
    }
    
    with open(bag_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        
        total_images = None
        summary = reader.get_summary()
        if summary and summary.statistics and summary.channels:
            total_images = 0
            for channel_id, channel in summary.channels.items():
                if channel.topic in topics and topics[channel.topic][1] == "image":
                    total_images += summary.statistics.channel_message_counts.get(channel_id, 0)
                    
        pbar = tqdm(total=total_images, desc="Preprocessing images")
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=list(topics.keys())):
            if channel.topic not in topics:
                continue
                
            out_dir, msg_type = topics[channel.topic]
            
            if msg_type == "info":
                camera_yaml_path = os.path.join(out_dir, "camera.yaml")
                if not os.path.exists(camera_yaml_path):
                    cam_params = {
                        "width": ros_msg.width,
                        "height": ros_msg.height,
                        "distortion_model": ros_msg.distortion_model,
                        "d": list(ros_msg.d),
                        "k": list(ros_msg.k),
                        "r": list(ros_msg.r),
                        "p": list(ros_msg.p)
                    }
                    with open(camera_yaml_path, "w") as yaml_f:
                        yaml.dump(cam_params, yaml_f)
                    print(f"Saved camera parameters to {camera_yaml_path}")
                    
            elif msg_type == "image":
                timestamp_ns = message.log_time
                np_arr = np.frombuffer(ros_msg.data, np.uint8)
                image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if image_bgr is not None:
                    img_path = os.path.join(out_dir, f"{timestamp_ns}.png")
                    cv2.imwrite(img_path, image_bgr)
                pbar.update(1)
        pbar.close()
        
    export_marker = os.path.join(base_dataset_dir, "export_completed.txt")
    with open(export_marker, "w") as f:
        f.write("Completed")
    
    print("Preprocessing completed.")

def main():
    parser = argparse.ArgumentParser(description="cuVSLAM Monocular Odometry on preprocessed mcap bag")
    parser.add_argument("--bag", type=str, default="data/10deg.mcap", help="Path to mcap bag file")
    parser.add_argument("--topic_prefix", type=str, default="/wide_angle_camera_front", help="Topic prefix for camera")
    parser.add_argument("--output", type=str, default=None, help="Output trajectory file (TUM format). Defaults to <bag_name>.txt")
    parser.add_argument("--workers", type=int, default=8, help="Number of background workers for preloading images")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(os.path.basename(args.bag))[0] + ".txt"

    print(f"Saving output trajectory to {args.output}")

    bag_name = os.path.splitext(os.path.basename(args.bag))[0]
    camera_name = args.topic_prefix.split('_')[-1]
    base_dataset_dir = os.path.join(os.path.dirname(args.bag), bag_name)
    dataset_dir = os.path.join(base_dataset_dir, camera_name)

    export_marker = os.path.join(base_dataset_dir, "export_completed.txt")
    if not os.path.exists(export_marker):
        preprocess_bag(args.bag, base_dataset_dir)

    camera_yaml_path = os.path.join(dataset_dir, "camera.yaml")
    if not os.path.exists(camera_yaml_path):
        print(f"Error: {camera_yaml_path} not found. Preprocessing may have failed.")
        return

    with open(camera_yaml_path, "r") as f:
        cam_params = yaml.safe_load(f)

    # 1. Setup Camera Rig
    cam = cuvslam.Camera()
    cam.size = [cam_params["width"], cam_params["height"]]
    cam.focal = [cam_params["k"][0], cam_params["k"][4]]
    cam.principal = [cam_params["k"][2], cam_params["k"][5]]
    
    distortion_model = cam_params["distortion_model"]
    d_coeffs = cam_params["d"]
    if distortion_model == "equidistant":
        cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Fisheye, d_coeffs)
    else:
        cam.distortion = cuvslam.Distortion(
            cuvslam.Distortion.Model.Brown,
            d_coeffs + [0]*(5 - len(d_coeffs)) if len(d_coeffs) < 5 else d_coeffs[:5]
        )

    cam.rig_from_camera = cuvslam.Pose(rotation=[0, 0, 0, 1], translation=[0, 0, 0])
    
    rig = cuvslam.Rig()
    rig.cameras = [cam]

    cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_observations_export=True,
        enable_final_landmarks_export=True,
        rectified_stereo_camera=False,
        odometry_mode=cuvslam.Tracker.OdometryMode.Mono
    )

    tracker = cuvslam.Tracker(rig, cfg)
    print("cuVSLAM Tracker initialized with odometry mode: Mono")

    # Setup rerun visualizer
    rr.init("cuVSLAM Visualizer", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    blueprint = rrb.Blueprint(
        rrb.TimePanel(state="collapsed"),
        rrb.Horizontal(
            column_shares=[0.5, 0.5],
            contents=[
                rrb.Spatial2DView(origin='world/camera_0'),
                rrb.Spatial3DView(origin='world')
            ]
        )
    )
    rr.send_blueprint(blueprint)

    # Get preprocessed image frames
    image_paths = glob.glob(os.path.join(dataset_dir, "*.png"))
    image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    if not image_paths:
        print(f"Error: No images found in {dataset_dir}.")
        return

    # Multithreaded Image Preloading
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    image_queue = queue.Queue(maxsize=100)

    def load_image(image_path):
        return cv2.imread(image_path, cv2.IMREAD_COLOR)

    def image_producer():
        try:
            for img_path in image_paths:
                timestamp_ns = int(os.path.splitext(os.path.basename(img_path))[0])
                future = executor.submit(load_image, img_path)
                image_queue.put((timestamp_ns, future))
        except Exception as e:
            print(f"Producer thread exception: {e}")

    producer_thread = threading.Thread(target=image_producer, daemon=True)
    producer_thread.start()

    trajectory = []
    
    with open(args.output, "w") as output_file:
        for frame_id in tqdm(range(len(image_paths)), desc="Tracking Frames"):
            q_timestamp_ns, image_future = image_queue.get()
            image_bgr = image_future.result()
            
            if image_bgr is None:
                print(f"Warning: Failed to read image at timestamp {q_timestamp_ns}")
                continue

            odom_pose_estimate, _ = tracker.track(q_timestamp_ns, [image_bgr])
            
            if odom_pose_estimate.world_from_rig is None:
                print(f"Warning: Failed to track frame {frame_id}")
                continue
                
            odom_pose = odom_pose_estimate.world_from_rig.pose
            trajectory.append(odom_pose.translation)
            
            # TUM format: timestamp(s) tx ty tz qx qy qz qw
            ts_sec = q_timestamp_ns / 1e9
            t = odom_pose.translation
            q = odom_pose.rotation
            output_file.write(f"{ts_sec:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
            
            # Visualize
            rr.set_time_sequence("frame", frame_id)
            rr.log("world/trajectory", rr.LineStrips3D(trajectory), static=True)
            rr.log(
                "world/camera_0",
                rr.Transform3D(
                    translation=odom_pose.translation,
                    quaternion=odom_pose.rotation
                ),
                rr.Arrows3D(
                    vectors=np.eye(3) * 0.2,
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
                )
            )

            current_observations_main_cam = tracker.get_last_observations(0)
            points = np.array([[obs.u, obs.v] for obs in current_observations_main_cam])
            colors = np.array([color_from_id(obs.id) for obs in current_observations_main_cam])
            
            # Convert BGR to RGB for correct display in rerun
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if len(points) > 0:
                rr.log(
                    "world/camera_0/observations",
                    rr.Points2D(positions=points, colors=colors, radii=5.0),
                    rr.Image(image_rgb).compress(jpeg_quality=80)
                )
            else:
                rr.log(
                    "world/camera_0/observations",
                    rr.Image(image_rgb).compress(jpeg_quality=80)
                )

    executor.shutdown(wait=False)
    print(f"Trajectory saved to {args.output}")

if __name__ == "__main__":
    main()
