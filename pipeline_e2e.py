#!/usr/bin/env python3
"""
End-to-end: Image → MapAnything → 3D Point Cloud → PointPillars

Usage:
    source /opt/pytorch/bin/activate
    cd ~/kitti-project
    python pipeline_e2e.py --idx 000008
"""

import sys, os, json, argparse, subprocess, tempfile
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Setup paths ──
sys.path.insert(0, '/home/ubuntu/map-anything')
sys.path.insert(0, '/home/ubuntu/pseudo_lidar/preprocessing')

KITTI = Path('/home/ubuntu/kitti-project/data/KITTI/object/training')
OUTDIR = Path('/home/ubuntu/kitti-project/output')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', default='000008')
    parser.add_argument('--score_thresh', type=float, default=0.3)
    args = parser.parse_args()
    idx = args.idx

    out = OUTDIR / f'e2e_{idx}'
    out.mkdir(parents=True, exist_ok=True)

    img_path = KITTI / f'image_2/{idx}.png'
    calib_path = KITTI / f'calib/{idx}.txt'
    print(f'\n=== Pipeline for sample {idx} ===\n')

    # ──────────────────────────────────────────────
    # STEP 1: Run MapAnything
    # ──────────────────────────────────────────────
    print('[1] Running MapAnything...')
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    model = MapAnything.from_pretrained('facebook/map-anything-apache').to('cuda')
    model.eval()

    views = load_images([str(img_path)])
    with torch.no_grad():
        preds = model.infer(views, use_amp=False, apply_mask=False,
                            mask_edges=False, apply_confidence_mask=False)
    pred = preds[0]

    pts3d = pred['pts3d'][0].cpu().numpy()              # (H, W, 3) — MapAnything camera frame
    mask = pred['non_ambiguous_mask'][0].cpu().numpy()   # (H, W) bool
    intrinsics = pred['intrinsics'][0].cpu().numpy()     # (3, 3) — MapAnything's intrinsics

    H_ma, W_ma = pts3d.shape[:2]
    pts_valid = pts3d.reshape(-1, 3)[mask.reshape(-1)]

    print(f'    pts3d shape: {pts3d.shape}')
    print(f'    valid points: {mask.sum():,} / {mask.size:,}')
    print(f'    MapAnything intrinsics: fx={intrinsics[0,0]:.1f}, fy={intrinsics[1,1]:.1f}')
    print(f'    depth range: {pts_valid[:,2].min():.1f} - {pts_valid[:,2].max():.1f}m')

    # Free GPU memory
    del model, views, preds, pred
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────
    # STEP 2: Render the raw 3D point cloud
    # ──────────────────────────────────────────────
    print('\n[2] Rendering 3D point cloud...')

    # Subsample for plotting
    if len(pts_valid) > 50000:
        idx_sub = np.random.choice(len(pts_valid), 50000, replace=False)
        pts_plot = pts_valid[idx_sub]
    else:
        pts_plot = pts_valid

    depth = pts_plot[:, 2]
    d_norm = np.clip(depth / 60, 0, 1)
    colors = plt.cm.turbo(d_norm)

    # 4 views of the raw point cloud
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#1a1a2e')

    views_3d = [
        ('Front (looking down Z)', 0, -90),
        ('Top-down (XZ plane)', 90, -90),
        ('Side (looking down X)', 0, 0),
        ('Isometric', 25, -60),
    ]

    for i, (title, elev, azim) in enumerate(views_3d):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.set_facecolor('#0d1117')
        ax.scatter(pts_plot[:, 0], pts_plot[:, 2], -pts_plot[:, 1],
                   c=colors, s=0.1, alpha=0.6, rasterized=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X (right)', color='white', fontsize=8)
        ax.set_ylabel('Z (forward)', color='white', fontsize=8)
        ax.set_zlabel('-Y (up)', color='white', fontsize=8)
        ax.set_title(title, color='white', fontsize=10, fontweight='bold')
        ax.tick_params(colors='#666', labelsize=6)

    fig.suptitle(f'MapAnything Raw Point Cloud — {len(pts_valid):,} points\n'
                 f'Camera frame: X=right, Y=down, Z=forward  |  Colored by depth',
                 color='white', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path_3d = out / '2_pointcloud_3d.png'
    fig.savefig(str(path_3d), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'    Saved: {path_3d}')

    # Also render depth map
    depth_img = pts3d[:, :, 2].copy()
    depth_img[~mask] = 0
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.patch.set_facecolor('#1a1a2e')
    img_rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image', color='white', fontweight='bold')
    im = axes[1].imshow(depth_img, cmap='turbo', vmin=0, vmax=80)
    axes[1].set_title(f'MapAnything Depth (Z) — {H_ma}x{W_ma}', color='white', fontweight='bold')
    plt.colorbar(im, ax=axes[1], fraction=0.046, label='meters')
    for ax in axes: ax.tick_params(colors='#888')
    plt.tight_layout()
    path_depth = out / '2_depth_map.png'
    fig.savefig(str(path_depth), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'    Saved: {path_depth}')

    # ──────────────────────────────────────────────
    # STEP 3: Convert to velodyne coordinates
    # ──────────────────────────────────────────────
    print('\n[3] Converting to velodyne coordinates...')
    import kitti_util

    calib = kitti_util.Calibration(str(calib_path))

    # pts3d is in MapAnything's camera frame (X right, Y down, Z forward)
    # Treat as rect camera coords and transform to velodyne
    pts_velo = calib.project_rect_to_velo(pts_valid)

    print(f'    Before filtering: {len(pts_velo):,} points')
    print(f'      x(fwd): [{pts_velo[:,0].min():.1f}, {pts_velo[:,0].max():.1f}]')
    print(f'      y(left): [{pts_velo[:,1].min():.1f}, {pts_velo[:,1].max():.1f}]')
    print(f'      z(up): [{pts_velo[:,2].min():.1f}, {pts_velo[:,2].max():.1f}]')

    # Filter: forward only + height cap
    keep = (pts_velo[:, 0] >= 0) & (pts_velo[:, 2] < 1.0)
    pts_velo = pts_velo[keep]

    print(f'    After filtering (x>=0, z<1): {len(pts_velo):,} points')

    # Add intensity = 1.0 and save as .bin
    cloud = np.hstack([pts_velo, np.ones((len(pts_velo), 1))]).astype(np.float32)
    bin_path = out / f'{idx}_pseudo.bin'
    cloud.tofile(str(bin_path))
    print(f'    Saved: {bin_path} ({cloud.shape[0]:,} points)')

    # Render BEV of the converted cloud
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0d1117')
    p = cloud[np.random.choice(len(cloud), min(80000, len(cloud)), replace=False)]
    ax.scatter(p[:, 1], p[:, 0], c='#ff8a65', s=0.1, alpha=0.5, edgecolors='none', rasterized=True)
    ax.add_patch(plt.Rectangle((-0.9, -2), 1.8, 4, lw=1.5, edgecolor='white', facecolor='#333', zorder=10))
    ax.set_xlim(-45, 45); ax.set_ylim(-5, 75); ax.set_aspect('equal')
    ax.set_xlabel('Y (left/right) [m]', color='white')
    ax.set_ylabel('X (forward) [m]', color='white')
    ax.set_title(f'Pseudo-LiDAR BEV — {len(cloud):,} pts', color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#888'); ax.grid(True, alpha=0.15, color='white'); ax.invert_xaxis()
    for s in ax.spines.values(): s.set_color('#333')
    path_bev = out / '3_bev.png'
    fig.savefig(str(path_bev), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'    Saved: {path_bev}')

    # ──────────────────────────────────────────────
    # STEP 4: Run PointPillars (in subprocess)
    # ──────────────────────────────────────────────
    print('\n[4] Running PointPillars...')

    inference_code = '''
import sys, os, json, numpy as np
sys.path.insert(0, '/home/ubuntu/OpenPCDet')
os.chdir('/home/ubuntu/OpenPCDet/tools')
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

class SF(DatasetTemplate):
    def __init__(self, dc, cn, bp, lg=None):
        super().__init__(dataset_cfg=dc, class_names=cn, training=False, root_path=None, logger=lg)
        self.bp = bp
    def __len__(self): return 1
    def __getitem__(self, i):
        return self.prepare_data({'points': np.fromfile(self.bp, dtype=np.float32).reshape(-1,4), 'frame_id': 0})

lg = common_utils.create_logger()
cfg_from_yaml_file('/home/ubuntu/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml', cfg)
ds = SF(cfg.DATA_CONFIG, cfg.CLASS_NAMES, sys.argv[1], lg)
m = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
m.load_params_from_file(filename='/home/ubuntu/OpenPCDet/pointpillar_7728.pth', logger=lg, to_cpu=True)
m.cuda(); m.eval()
with torch.no_grad():
    d = ds[0]; d = ds.collate_batch([d]); load_data_to_gpu(d)
    p, _ = m.forward(d)
bx, sc, lb = p[0]['pred_boxes'].cpu().numpy(), p[0]['pred_scores'].cpu().numpy(), p[0]['pred_labels'].cpu().numpy()
mk = sc >= float(sys.argv[2])
json.dump({'boxes': bx[mk].tolist(), 'scores': sc[mk].tolist(), 'labels': lb[mk].tolist()}, open(sys.argv[3],'w'))
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(inference_code); script = f.name
    out_json = str(out / 'pp_result.json')

    result = subprocess.run(
        [sys.executable, script, str(bin_path), str(args.score_thresh), out_json],
        capture_output=True, text=True, timeout=120
    )
    os.unlink(script)

    if result.returncode != 0:
        print(f'    ERROR: {result.stderr[-500:]}')
        return

    with open(out_json) as f:
        det = json.load(f)

    boxes = np.array(det['boxes']).reshape(-1, 7) if det['boxes'] else np.zeros((0, 7))
    scores = np.array(det['scores']) if det['scores'] else np.zeros(0)
    labels = np.array(det['labels'], dtype=np.int32) if det['labels'] else np.zeros(0, dtype=np.int32)

    print(f'    Detections (score >= {args.score_thresh}): {len(boxes)}')
    if len(scores) > 0:
        print(f'    Scores: {np.round(scores, 2)}')

    # ──────────────────────────────────────────────
    # STEP 5: Draw results on image
    # ──────────────────────────────────────────────
    print('\n[5] Drawing results on image...')
    img = cv2.imread(str(img_path))
    CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']

    # Draw GT (green)
    img_out = img.copy()
    with open(KITTI / f'label_2/{args.idx}.txt') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == 'DontCare': continue
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            center_rect = np.array([[x, y - h/2, z]])
            center_velo = calib.project_rect_to_velo(center_rect)[0]
            heading = -(ry + np.pi / 2)
            _draw_box(img_out, center_velo, l, w, h, heading, calib, (0, 255, 0), 2)

    # Draw detections (red)
    for i in range(len(boxes)):
        x, y, z, dx, dy, dz, hd = boxes[i]
        _draw_box(img_out, np.array([x, y, z]), dx, dy, dz, hd, calib, (0, 100, 255), 2)
        # Score label
        corners = _box_corners(np.array([x, y, z]), dx, dy, dz, hd)
        p2d = calib.project_rect_to_image(calib.project_velo_to_rect(corners)).astype(np.int32)
        my = np.argmin(p2d[:, 1])
        lbl = CLASS_NAMES[labels[i] - 1] if labels[i] <= len(CLASS_NAMES) else '?'
        cv2.putText(img_out, f'{lbl} {scores[i]:.2f}', (int(p2d[my, 0]), int(p2d[my, 1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

    # Add label banner
    banner = np.zeros((50, img_out.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, f'Green=GT  |  Red=PointPillars on Pseudo-LiDAR ({len(boxes)} det, thresh={args.score_thresh})',
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    result_img = np.vstack([banner, img_out])
    path_result = out / '5_result.png'
    cv2.imwrite(str(path_result), result_img)
    print(f'    Saved: {path_result}')

    # ── Done ──
    print(f'\n=== Done ===')
    print(f'Output directory: {out}/')
    for f in sorted(out.iterdir()):
        if f.suffix == '.json': continue
        print(f'  {f.name:40s} ({f.stat().st_size/1024:.0f} KB)')
    print(f'\nscp -r -i ~/.ssh/kitti-project.pem ubuntu@<IP>:{out} .')


def _box_corners(center, dx, dy, dz, heading):
    t = np.array([[1,1,-1],[1,-1,-1],[-1,-1,-1],[-1,1,-1],
                   [1,1,1],[1,-1,1],[-1,-1,1],[-1,1,1]]) / 2.0
    corners = t * np.array([dx, dy, dz])
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return corners @ rot.T + center


def _draw_box(img, center_velo, dx, dy, dz, heading, calib, color, thickness):
    corners = _box_corners(center_velo, dx, dy, dz, heading)
    cr = calib.project_velo_to_rect(corners)
    p2d = calib.project_rect_to_image(cr).astype(np.int32)
    h, w = img.shape[:2]
    if np.all(p2d[:, 0] < 0) or np.all(p2d[:, 0] > w): return
    if np.all(p2d[:, 1] < 0) or np.all(p2d[:, 1] > h): return
    for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        cv2.line(img, tuple(p2d[a]), tuple(p2d[b]), color, thickness)


if __name__ == '__main__':
    main()