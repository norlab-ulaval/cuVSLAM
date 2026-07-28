import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot exported trajectories")
    parser.add_argument("--sequence_dir", type=str, default="output", help="Path to sequence dir (containing localization trajectories)")
    parser.add_argument("--slam_traj", type=str, help="Path to the original SLAM trajectory (e.g. from remote evaluation folder)")
    args = parser.parse_args()

    files = {
        "Localization (Odom)": os.path.join(args.sequence_dir, "trajectory_odom_tum.txt"),
        "Localization (SLAM)": os.path.join(args.sequence_dir, "trajectory_tum.txt")
    }

    if args.slam_traj:
        files["Original SLAM"] = args.slam_traj

    plt.figure(figsize=(10, 8))

    for label, filepath in files.items():
        if os.path.exists(filepath):
            data = np.loadtxt(filepath)
            if len(data) > 0:
                # TUM format: timestamp, tx, ty, tz, qx, qy, qz, qw
                tx = data[:, 1]
                tz = data[:, 3]  # Z is usually forward in camera coordinates
                plt.plot(tx, tz, label=label, linewidth=2, alpha=0.8)
                
                # Mark start and end points
                plt.scatter(tx[0], tz[0], marker='o', s=50, zorder=5)
                plt.scatter(tx[-1], tz[-1], marker='x', s=50, zorder=5)
            else:
                print(f"Warning: {filepath} is empty.")
        else:
            print(f"Warning: {filepath} not found.")

    plt.title("Trajectory Comparison")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    output_png = os.path.join(args.sequence_dir, "trajectory_plot.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_png}")

if __name__ == "__main__":
    main()
