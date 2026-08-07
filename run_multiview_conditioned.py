#!/usr/bin/env python3
"""
run_multiview_conditioned.py — Run MapAnything with KITTI intrinsics + OXTS poses.
Outputs per-frame point clouds in camera rect coords.

Usage:
    python run_multiview_conditioned.py --seq 0 --start 2 --end 152 --window 4
    python run_multiview_conditioned.py --seq 0 --start 2 --end 152 --window 4 --save_rrd output.rrd
"""

import os, sys, time, argparse
import numpy as np, torch
from pathlib import Path
from math import cos, sin, radians
from PIL import Image

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.expanduser("~/map-anything"))

ROOT = Path("~/kitti-project/data/KITTI/tracking/training").expanduser()
OUT = Path("~/video_pipeline/outputs/conditioned").expanduser()


def parse_calib(p):
    c = {}
    for line in open(p):
        line = line.strip()
        if not line: continue
        if ":" in line: k, v = line.split(":", 1)
        else: parts = line.split(); k, v = parts[0], " ".join(parts[1:])
        c[k.strip()] = np.array([float(x) for x in v.split()])
    return c


def load_oxts_all(path):
    poses = []
    with open(path) as f:
        for line in f:
            poses.append(list(map(float, line.strip().split())))
    return poses


def compute_camera_pose(oxts, ref_lat, ref_lon, ref_alt, Tr_imu_velo, Tr_velo_cam, R_rect):
    lat, lon, alt = oxts[0], oxts[1], oxts[2]
    roll, pitch, yaw = oxts[3], oxts[4], oxts[5]

    er = 6378137.0
    scale = cos(radians(ref_lat))
    tx = scale * radians(lon - ref_lon) * er
    ty = scale * radians(lat - ref_lat) * er
    tz = alt - ref_alt

    Rx = np.array([[1,0,0],[0,cos(roll),-sin(roll)],[0,sin(roll),cos(roll)]])
    Ry = np.array([[cos(pitch),0,sin(pitch)],[0,1,0],[-sin(pitch),0,cos(pitch)]])
    Rz = np.array([[cos(yaw),-sin(yaw),0],[sin(yaw),cos(yaw),0],[0,0,1]])

    T = np.eye(4)
    T[:3,:3] = Rz @ Ry @ Rx
    T[:3,3] = [tx, ty, tz]

    return T @ np.linalg.inv(Tr_imu_velo) @ np.linalg.inv(Tr_velo_cam) @ np.linalg.inv(R_rect)


def reproject_ma(pts3d_cam, mask, K, img_h, img_w):
    h, w = pts3d_cam.shape[:2]
    depth = pts3d_cam[:,:,2].copy()
    sx, sy = img_w / w, img_h / h
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    X = ((uu+0.5)*sx - 0.5 - K[0,2]) * depth / K[0,0]
    Y = ((vv+0.5)*sy - 0.5 - K[1,2]) * depth / K[1,1]
    return np.stack([X, Y, depth], axis=-1)


def cam_to_rerun(pts):
    return np.stack([pts[:,0], pts[:,2], -pts[:,1]], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=0)
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--save_rrd", type=str, default=None)
    parser.add_argument("--save_npy", action="store_true",
                        help="Save per-frame .npy point clouds")
    args = parser.parse_args()

    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_id = f"{args.seq:04d}"
    img_dir = ROOT / "image_02" / seq_id
    velo_dir = ROOT / "velodyne" / seq_id
    calib_path = ROOT / "calib" / f"{seq_id}.txt"
    oxts_path = ROOT / "oxts" / f"{seq_id}.txt"

    # Calib
    raw = parse_calib(calib_path)
    P2 = raw["P2"].reshape(3, 4)
    R_rect = np.eye(4); R_rect[:3,:3] = raw["R_rect"].reshape(3,3)
    Tr_velo_cam = np.eye(4); Tr_velo_cam[:3,:4] = raw["Tr_velo_cam"].reshape(3,4)
    Tr_imu_velo = np.eye(4); Tr_imu_velo[:3,:4] = raw["Tr_imu_velo"].reshape(3,4)
    K = np.eye(3)
    K[0,0], K[1,1], K[0,2], K[1,2] = P2[0,0], P2[1,1], P2[0,2], P2[1,2]
    K_tensor = torch.tensor(K, dtype=torch.float32).unsqueeze(0).to(device)

    # OXTS
    all_oxts = load_oxts_all(oxts_path)
    ref_lat, ref_lon, ref_alt = all_oxts[0][0], all_oxts[0][1], all_oxts[0][2]

    # Model
    print(f"Loading MapAnything...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.eval()
    print(f"Loaded. Device={device}\n")

    # Frame range
    all_frames = sorted(img_dir.glob("*.png"))
    n = len(all_frames)
    half_w = args.window // 2
    start = max(args.start, half_w)
    end = min(args.end if args.end > 0 else n, n - half_w)

    print(f"Seq {seq_id}: frames {start}-{end}, window={args.window}, every={args.every}")
    print(f"Intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"OXTS poses: {len(all_oxts)} frames\n")

    # Rerun
    use_rerun = args.save_rrd is not None
    if use_rerun:
        import rerun as rr
        rr.init(f"Conditioned MapAnything — Seq {seq_id}")
        rr.save(args.save_rrd)

    # NPY output
    if args.save_npy:
        npy_dir = OUT / seq_id / "pts3d"
        os.makedirs(npy_dir, exist_ok=True)

    os.makedirs(OUT / seq_id, exist_ok=True)

    for fi in range(start, end, args.every):
        fid = f"{fi:06d}"
        t0 = time.time()

        # Build window
        win_indices = list(range(fi - half_w, fi + half_w))
        win_paths = [str(img_dir / f"{i:06d}.png") for i in win_indices]

        views = load_images(win_paths)

        # Inject intrinsics
        for v in views:
            v["intrinsics"] = K_tensor

        # Inject poses
        for vi, idx in enumerate(win_indices):
            pose = compute_camera_pose(all_oxts[idx], ref_lat, ref_lon, ref_alt,
                                       Tr_imu_velo, Tr_velo_cam, R_rect)
            views[vi]["camera_poses"] = torch.tensor(pose, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model.infer(views, use_amp=True, amp_dtype="bf16",
                                apply_mask=False, mask_edges=False,
                                apply_confidence_mask=False,
                                memory_efficient_inference=True,
                                use_multiview_confidence=True)

        # Extract center frame
        center = fi - win_indices[0]
        pts3d_raw = preds[center]["pts3d_cam"].squeeze(0).cpu().numpy()
        mask = preds[center]["non_ambiguous_mask"].squeeze(0).cpu().numpy() > 0
        conf = preds[center]["conf"].squeeze(0).cpu().numpy()

        # Reproject through KITTI intrinsics
        img = Image.open(img_dir / f"{fid}.png")
        img_w, img_h = img.size
        pts3d_kitti = reproject_ma(pts3d_raw, mask, K, img_h, img_w)

        elapsed = time.time() - t0

        # Save NPY
        if args.save_npy:
            np.save(npy_dir / f"{fid}_pts3d.npy", pts3d_kitti)
            np.save(npy_dir / f"{fid}_mask.npy", mask)
            np.save(npy_dir / f"{fid}_conf.npy", conf)

        # Rerun logging
        if use_rerun:
            import rerun as rr
            rr.set_time("frame", sequence=fi)

            img_np = np.array(img)
            rr.log("camera/image", rr.Image(img_np))

            # MapAnything (blue)
            pts_flat = pts3d_kitti.reshape(-1,3)[mask.reshape(-1)]
            pts_flat = pts_flat[pts_flat[:,2] > 0.5]
            blue = np.full((len(pts_flat),3), [60,100,230], dtype=np.uint8)
            rr.log("world/mapanything", rr.Points3D(cam_to_rerun(pts_flat), colors=blue, radii=0.04))

            # Real LiDAR (green)
            velo = np.fromfile(str(velo_dir / f"{fid}.bin"), dtype=np.float32).reshape(-1,4)
            pts_h = np.concatenate([velo[:,:3], np.ones((len(velo),1))], axis=1)
            pts_rect = (R_rect @ Tr_velo_cam @ pts_h.T).T[:,:3]
            pts_rect = pts_rect[pts_rect[:,2] > 0.5]
            green = np.full((len(pts_rect),3), [50,220,50], dtype=np.uint8)
            rr.log("world/real_lidar", rr.Points3D(cam_to_rerun(pts_rect), colors=green, radii=0.04))

        if fi % 10 == 0 or fi == start:
            depth_med = np.median(pts3d_kitti[:,:,2][mask])
            print(f"  Frame {fid}: {mask.sum():,} valid pts, "
                  f"depth median={depth_med:.1f}m, {elapsed:.2f}s")

        torch.cuda.empty_cache()

    print(f"\nDone!")
    if args.save_npy:
        print(f"  NPY files: {npy_dir}")
    if use_rerun:
        print(f"  Rerun: {args.save_rrd}")

if __name__ == "__main__":
    main()
