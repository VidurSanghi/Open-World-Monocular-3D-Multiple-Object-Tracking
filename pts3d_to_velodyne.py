import argparse
import os
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Convert MapAnything pts3d to pseudo-LiDAR .bin files')
    parser.add_argument('--pts3d_dir', required=True, help='Directory containing _pts3d.npy and _mask.npy files')
    parser.add_argument('--calib_dir', required=True, help='Directory containing KITTI calib .txt files')
    parser.add_argument('--save_dir', required=True, help='Output directory for .bin point clouds')
    parser.add_argument('--split_file', default=None, help='Optional split file to limit which files to process')
    parser.add_argument('--max_high', type=float, default=1.0, help='Max height filter (meters)')
    args = parser.parse_args()

    sys.path.insert(0, os.path.expanduser('~/pseudo_lidar/preprocessing'))
    import kitti_util

    os.makedirs(args.save_dir, exist_ok=True)

    if args.split_file:
        with open(args.split_file) as f:
            stems = [line.strip() for line in f if line.strip()]
    else:
        stems = [f.replace('_pts3d.npy', '') for f in sorted(os.listdir(args.pts3d_dir)) if f.endswith('_pts3d.npy')]

    print(f"Processing {len(stems)} files...")
    success, failed = 0, 0

    for stem in stems:
        pts3d_path = os.path.join(args.pts3d_dir, stem + '_pts3d.npy')
        mask_path = os.path.join(args.pts3d_dir, stem + '_mask.npy')
        calib_path = os.path.join(args.calib_dir, stem + '.txt')
        save_path = os.path.join(args.save_dir, stem + '.bin')

        if not os.path.exists(pts3d_path):
            print(f"SKIP {stem}: no pts3d file")
            failed += 1
            continue

        try:
            # pts3d is (H, W, 3) in rect camera coord (X right, Y down, Z forward)
            pts3d = np.load(pts3d_path)  # (H, W, 3)
            mask = np.load(mask_path)    # (H, W) bool

            # Flatten and apply mask
            pts = pts3d.reshape(-1, 3)
            pts = pts[mask.reshape(-1)]

            # Transform rect camera coords -> velodyne coords using kitti_util
            calib = kitti_util.Calibration(calib_path)
            cloud = calib.project_rect_to_velo(pts)  # (N, 3)

            # Filter: points in front of sensor and below max_high
            valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < args.max_high)
            cloud = cloud[valid]

            # Append intensity=1 (standard pseudo-lidar convention)
            cloud = np.concatenate([cloud, np.ones((cloud.shape[0], 1))], axis=1)
            cloud = cloud.astype(np.float32)
            cloud.tofile(save_path)

            success += 1
            if success % 100 == 0:
                print(f"Progress: {success}/{len(stems)}")

        except Exception as e:
            import traceback
            print(f"FAIL {stem}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nDone: {success} success, {failed} failed out of {len(stems)}")

if __name__ == '__main__':
    main()