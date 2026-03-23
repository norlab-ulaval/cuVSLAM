import os
import argparse
import logging
import numpy as np

import cuvslam
from dataset_custom import get_custom_rig, prepare_custom_metadata, load_frame

def main():
    parser = argparse.ArgumentParser(description="PyCuVSLAM Custom Data Tracker")
    parser.add_argument(
        '--data', type=str, default='/home/mbo/pycuVSLAM/example-data',
        help='Path to the custom dataset directory.'
    )
    parser.add_argument('--map-out', type=str, default=None, help='Folder where to save the map output.')
    parser.add_argument('--traj-out', type=str, default=None, help='Full filepath where to save the .txt trajectory file.')
    parser.add_argument('--enable-slam', action='store_true', help='Enable SLAM to perform global loop closures.')
    args = parser.parse_args()

    # Configure Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Constants
    data_dir = args.data

    # Setup tracking run specifics
    tracking_mode = cuvslam.Tracker.OdometryMode(1)  # OdometryMode.Inertial
    logger.info(f"Starting PyCuVSLAM Tracker in mode: {tracking_mode}")

    # Build Configuration
    cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_observations_export=True,
        enable_final_landmarks_export=True,
        horizontal_stereo_camera=False, # ZED is normally horizontal but this toggles a specific mode
        odometry_mode=tracking_mode
    )

    # Prepare Rig & Tracker
    logger.info("Initializing Custom Rig parameters...")
    rig = get_custom_rig(data_dir)
    if args.enable_slam:
        s_cfg = cuvslam.Tracker.SlamConfig(sync_mode=False)
        tracker = cuvslam.Tracker(rig, cfg, s_cfg)
        logger.info("SLAM enabled.")
    else:
        tracker = cuvslam.Tracker(rig, cfg)
    
    # Load Metadatas
    logger.info("Parsing event metadatas logically...")
    events = prepare_custom_metadata(data_dir)
    logger.info(f"Extracted {len(events)} chronological events.")

    # Tracking Loop Variables
    frame_id = 0
    odom_trajectory = [] # [timestamp_sec, tx,ty,tz, qx,qy,qz,qw]
    slam_trajectory = [] # [timestamp_sec, tx,ty,tz, qx,qy,qz,qw]
    last_cam_time = None
    imu_count = 0
    lost_track_count = 0

    logger.info("Commencing Tracker Loop...")

    for event in events:
        ts_ns = event['timestamp']

        if event['type'] == 'imu':
            imu_meas = cuvslam.ImuMeasurement()
            imu_meas.timestamp_ns = ts_ns
            # Ensure properties are mapped right based on vectors format:
            # According to setup: [ax, ay, az] and [lx, ly, lz(gyro)]
            imu_meas.linear_accelerations = np.asarray(event['accel'], dtype=np.float64)
            imu_meas.angular_velocities = np.asarray(event['gyro'], dtype=np.float64)
            tracker.register_imu_measurement(0, imu_meas)
            imu_count += 1
            continue

        elif event['type'] == 'stereo':
            # Pre-Track checks
            if last_cam_time is not None and imu_count == 0:
                logger.warning(f"No IMU packets synced between frame {frame_id}! Last IMU Count 0")
            
            # Reset counters
            last_cam_time = ts_ns
            imu_count = 0

            imgs_np = [load_frame(p) for p in event['images_paths']]
            
            # Submit to PyCuVSLAM
            odom_pose_estimate, slam_pose = tracker.track(ts_ns, imgs_np)

            if odom_pose_estimate.world_from_rig is None:
                lost_track_count += 1
                logger.warning(f"Frame {frame_id} lost tracking pose estimation. Total lost: {lost_track_count}")
                if lost_track_count > 100:
                    logger.error("Lost track for more than 100 total frames. Exiting.")
                    break
            else:
                pose = odom_pose_estimate.world_from_rig.pose
                
                # IMPORTANT: Storing trajectory timestamp in SECONDS
                sec_epoch = ts_ns / 1_000_000_000.0
                
                odom_trajectory.append([
                    sec_epoch,
                    pose.translation[0], pose.translation[1], pose.translation[2],
                    pose.rotation[0], pose.rotation[1], pose.rotation[2], pose.rotation[3]
                ])

                # If SLAM is enabled and provides a pose, save it as well.
                # In PyCuVSLAM, slam_pose could be provided when SLAM completes updates.
                if slam_pose is not None:
                    slam_trajectory.append([
                        sec_epoch,
                        slam_pose.translation[0], slam_pose.translation[1], slam_pose.translation[2],
                        slam_pose.rotation[0], slam_pose.rotation[1], slam_pose.rotation[2], slam_pose.rotation[3]
                    ])

            # Iteration Logging Updates
            if frame_id % 100 == 0:
                logger.info(f"Processed {frame_id} keyframes successfully...")
            frame_id += 1

    # End state, export data maps
    logger.info("Sequence completed. Exporting trajectory targets...")
    
    # Export TUM Style 
    traj_out = args.traj_out if args.traj_out else os.path.join(data_dir, "trajectory.txt")
    traj_out.replace(".txt", ".txt_bak")
    if odom_trajectory:
        np.savetxt(traj_out, odom_trajectory, delimiter=' ', fmt='%f')
        logger.info(f"Odom Trajectory dumped to: {traj_out}")
    else:
        logger.warning(f"No odom trajectory to dump: {traj_out}")

    if args.enable_slam:
        slam_traj_out = traj_out.replace(".txt_bak", ".txt")
        if slam_trajectory:
            np.savetxt(slam_traj_out, slam_trajectory, delimiter=' ', fmt='%f')
            logger.info(f"SLAM Trajectory dumped to: {slam_traj_out}")
        else:
            logger.warning(f"No slam trajectory to dump: {slam_traj_out}")
    
    # Save cuvslam Map state
    map_out = args.map_out if args.map_out else os.path.join(data_dir, "map_output.cuvslam")
    
    def save_callback(status):
        logger.info(f"Map save callback triggered with status: {status}")

    if hasattr(tracker, 'save_map'):
        tracker.save_map(map_out, save_callback)
        logger.info(f"Map Export requested to: {map_out}")
    elif hasattr(tracker.odom, 'save_map'):
        tracker.odom.save_map(map_out, save_callback)
        logger.info(f"Map Export requested to: {map_out}")
    else:
        logger.info(f"save_map target missing from Tracker, PyCuVSLAM interface might differ locally.")
        
    logger.info("Tracking Process Finished Successfully.")

if __name__ == "__main__":
    main()
