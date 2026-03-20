#!/usr/bin/env python3
"""
End-to-End Pipeline Debug: Image → MapAnything → Point Cloud → PointPillars

Generates diagnostic visualizations at every step to inspect the pipeline.

Usage:
    source /opt/pytorch/bin/activate
    cd ~/kitti-project
    python pipeline_debug.py --idx 000008
    python pipeline_debug.py --idx 000008 --skip_inference   # faster, skip PointPillars

Output: ~/kitti-project/output/pipeline_{idx}/
"""

import sys
import os
import json
import argparse
import subprocess
import tempfile
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

sys.path.insert(0, os.path.expanduser('~/pseudo_lidar/preprocessing'))
import kitti_util

# ── Paths ──
KITTI_ROOT = Path('/home/ubuntu/kitti-project/data/KITTI/object/training')
PTS3D_DIR  = Path('/home/ubuntu/kitti-project/pts3d_output')
OUTPUT_DIR = Path('/home/ubuntu/kitti-project/output')


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def subsample(pts, n=50000):
    if len(pts) <= n:
        return pts
    return pts[np.random.choice(len(pts), n, replace=False)]


def savefig(fig, path, dpi=150):
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  → {path.name}')


def parse_gt_boxes_velo(label_path, calib):
    """Parse GT labels and return list of boxes in velodyne coords."""
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            cls = parts[0]
            if cls == 'DontCare':
                continue
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            center_rect = np.array([[x, y - h / 2, z]])
            center_velo = calib.project_rect_to_velo(center_rect)[0]
            heading = -(ry + np.pi / 2)
            dist = np.sqrt(center_velo[0]**2 + center_velo[1]**2)
            boxes.append({
                'cls': cls, 'h': h, 'w': w, 'l': l,
                'center_velo': center_velo, 'heading': heading, 'dist': dist,
            })
    return boxes


def count_points_in_box(pts, box):
    """Count points inside an oriented 3D box."""
    dx = pts[:, :3] - box['center_velo']
    cos_h, sin_h = np.cos(-box['heading']), np.sin(-box['heading'])
    rx = dx[:, 0] * cos_h - dx[:, 1] * sin_h
    ry = dx[:, 0] * sin_h + dx[:, 1] * cos_h
    rz = dx[:, 2]
    inside = (np.abs(rx) < box['l'] / 2) & (np.abs(ry) < box['w'] / 2) & (np.abs(rz) < box['h'] / 2)
    return inside.sum()


def draw_gt_bev(ax, gt_boxes, color='#00ff88'):
    for b in gt_boxes:
        c = b['center_velo']
        heading = b['heading']
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        dx, dy = b['l'] / 2, b['w'] / 2
        corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        corners = (rot @ corners.T).T + c[:2]
        polygon = plt.Polygon(corners, fill=False, edgecolor=color,
                               linewidth=1.5, linestyle='--', zorder=5)
        ax.add_patch(polygon)
        ax.text(c[0], c[1], f"{b['cls']}\n{b['dist']:.0f}m",
                fontsize=6, color=color, ha='center', va='center', zorder=6,
                fontweight='bold', path_effects=[pe.withStroke(linewidth=2, foreground='black')])


# ═══════════════════════════════════════════════════════════════════
# PointPillars subprocess inference
# ═══════════════════════════════════════════════════════════════════

INFERENCE_SCRIPT = '''
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
        pts = np.fromfile(self.bp, dtype=np.float32).reshape(-1, 4)
        return self.prepare_data({'points': pts, 'frame_id': 0})

bp, st, op = sys.argv[1], float(sys.argv[2]), sys.argv[3]
lg = common_utils.create_logger()
cfg_from_yaml_file('/home/ubuntu/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml', cfg)
ds = SF(cfg.DATA_CONFIG, cfg.CLASS_NAMES, bp, lg)
m = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
m.load_params_from_file(filename='/home/ubuntu/OpenPCDet/pointpillar_7728.pth', logger=lg, to_cpu=True)
m.cuda(); m.eval()
with torch.no_grad():
    d = ds[0]; d = ds.collate_batch([d]); load_data_to_gpu(d)
    p, _ = m.forward(d)
bx, sc, lb = p[0]['pred_boxes'].cpu().numpy(), p[0]['pred_scores'].cpu().numpy(), p[0]['pred_labels'].cpu().numpy()
mk = sc >= st
json.dump({'boxes': bx[mk].tolist(), 'scores': sc[mk].tolist(), 'labels': lb[mk].tolist()}, open(op, 'w'))
'''


def run_pp(bin_path, thresh=0.3):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as sf:
        sf.write(INFERENCE_SCRIPT); sp = sf.name
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as of:
        op = of.name
    try:
        r = subprocess.run([sys.executable, sp, str(bin_path), str(thresh), op],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f'  ERR: {r.stderr[-300:]}')
            return np.zeros((0, 7)), np.zeros(0), np.zeros(0, dtype=np.int32)
        d = json.load(open(op))
        bx = np.array(d['boxes']).reshape(-1, 7) if d['boxes'] else np.zeros((0, 7))
        sc = np.array(d['scores']) if d['scores'] else np.zeros(0)
        lb = np.array(d['labels'], dtype=np.int32) if d['labels'] else np.zeros(0, dtype=np.int32)
        return bx, sc, lb
    finally:
        os.unlink(sp); os.unlink(op)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=str, default='000008')
    parser.add_argument('--score_thresh', type=float, default=0.3)
    parser.add_argument('--skip_inference', action='store_true')
    args = parser.parse_args()

    idx = args.idx
    out = OUTPUT_DIR / f'pipeline_{idx}'
    out.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*70}')
    print(f'  Pipeline Debug: Sample {idx}')
    print(f'{"="*70}')

    # ── Load everything ──
    img = cv2.imread(str(KITTI_ROOT / f'image_2/{idx}.png'))
    calib = kitti_util.Calibration(str(KITTI_ROOT / f'calib/{idx}.txt'))
    pts3d = np.load(str(PTS3D_DIR / f'{idx}_pts3d.npy'))   # (H, W, 3)
    mask = np.load(str(PTS3D_DIR / f'{idx}_mask.npy'))       # (H, W)
    real = np.fromfile(str(KITTI_ROOT / f'velodyne/{idx}.bin'), dtype=np.float32).reshape(-1, 4)
    pseudo = np.fromfile(str(KITTI_ROOT / f'pseudo-lidar_velodyne/{idx}.bin'), dtype=np.float32).reshape(-1, 4)
    gt_boxes = parse_gt_boxes_velo(str(KITTI_ROOT / f'label_2/{idx}.txt'), calib)

    H_ma, W_ma = pts3d.shape[:2]
    H_k, W_k = img.shape[:2]
    pts_valid = pts3d.reshape(-1, 3)[mask.reshape(-1)]

    print(f'  Image: {W_k}x{H_k}')
    print(f'  MapAnything: {W_ma}x{H_ma}, {mask.sum():,} valid pts')
    print(f'  Real LiDAR: {len(real):,} pts')
    print(f'  Pseudo-LiDAR .bin: {len(pseudo):,} pts')
    print(f'  GT boxes: {len(gt_boxes)}')

    # ════════════════════════════════════════════════════════════════
    # STEP 1: Original image with GT overlay
    # ════════════════════════════════════════════════════════════════
    print(f'\n[Step 1] Original image + GT boxes...')
    img_gt = img.copy()
    for b in gt_boxes:
        center = b['center_velo']
        heading = b['heading']
        template = np.array([[1,1,-1],[1,-1,-1],[-1,-1,-1],[-1,1,-1],
                              [1,1,1],[1,-1,1],[-1,-1,1],[-1,1,1]]) / 2.0
        corners = template * np.array([b['l'], b['w'], b['h']])
        c, s = np.cos(heading), np.sin(heading)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        corners = corners @ rot.T + center
        corners_rect = calib.project_velo_to_rect(corners)
        pts_2d = calib.project_rect_to_image(corners_rect).astype(np.int32)
        for a, b_idx in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            cv2.line(img_gt, tuple(pts_2d[a]), tuple(pts_2d[b_idx]), (0, 255, 0), 2)
    cv2.imwrite(str(out / '1_image_gt.png'), img_gt)
    savefig(plt.figure(), out / '_dummy.png')  # just to not error
    print(f'  → 1_image_gt.png')

    # ════════════════════════════════════════════════════════════════
    # STEP 2: MapAnything depth map
    # ════════════════════════════════════════════════════════════════
    print(f'\n[Step 2] Depth map from MapAnything...')
    depth = pts3d[:, :, 2].copy()
    depth[~mask] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    fig.patch.set_facecolor('#1a1a2e')

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original Image ({W_k}x{H_k})', color='white', fontweight='bold')

    im = axes[1].imshow(depth, cmap='turbo', vmin=0, vmax=80)
    axes[1].set_title(f'Depth (Z fwd) — {H_ma}x{W_ma}\nrange: {np.nanmin(depth):.1f}-{np.nanmax(depth):.1f}m',
                       color='white', fontweight='bold')
    plt.colorbar(im, ax=axes[1], fraction=0.046, label='meters')

    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title(f'Valid Mask — {mask.sum():,}/{mask.size:,} ({100*mask.sum()/mask.size:.0f}%)',
                       color='white', fontweight='bold')

    for ax in axes:
        ax.tick_params(colors='#888888')
    plt.tight_layout()
    savefig(fig, out / '2_depth_map.png')

    # ════════════════════════════════════════════════════════════════
    # STEP 3: Points projected back onto image
    # ════════════════════════════════════════════════════════════════
    print(f'\n[Step 3] Points reprojected onto image...')
    # Project pseudo-LiDAR velodyne points back to image
    pts_rect = calib.project_velo_to_rect(pseudo[:, :3])
    pts_2d = calib.project_rect_to_image(pts_rect)
    depths = pts_rect[:, 2]

    in_img = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W_k) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H_k) & (depths > 0)
    pts_2d_valid = pts_2d[in_img]
    depths_valid = depths[in_img]

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    sub_idx = np.random.choice(len(pts_2d_valid), min(30000, len(pts_2d_valid)), replace=False)
    sc = ax.scatter(pts_2d_valid[sub_idx, 0], pts_2d_valid[sub_idx, 1],
                     c=depths_valid[sub_idx], cmap='turbo', s=0.3, alpha=0.7,
                     vmin=0, vmax=60, edgecolors='none', rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.03, label='Depth (m)')
    ax.set_title(f'Pseudo-LiDAR Points Projected on Image — {in_img.sum():,} pts visible',
                  color='white', fontsize=13, fontweight='bold')
    ax.set_xlim(0, W_k); ax.set_ylim(H_k, 0)
    ax.tick_params(colors='#888888')
    plt.tight_layout()
    savefig(fig, out / '3_projected_on_image.png')

    # ════════════════════════════════════════════════════════════════
    # STEP 4: 3D point cloud renders (rect camera coords)
    # ════════════════════════════════════════════════════════════════
    print(f'\n[Step 4] 3D renders in rect camera coords...')
    pts_sub = subsample(pts_valid, 40000)
    depth_color = pts_sub[:, 2]  # Z = forward depth
    d_norm = np.clip(depth_color / 60, 0, 1)

    for name, elev, azim in [('front', 0, -90), ('top', 90, -90), ('iso', 25, -60)]:
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a2e')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#0d1117')
        ax.scatter(pts_sub[:, 0], pts_sub[:, 2], pts_sub[:, 1],
                   c=plt.cm.turbo(d_norm), s=0.1, alpha=0.6, rasterized=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X (right)', color='white', fontsize=9)
        ax.set_ylabel('Z (forward)', color='white', fontsize=9)
        ax.set_zlabel('Y (down)', color='white', fontsize=9)
        ax.set_title(f'Rect Camera Coords — {name} view\n{len(pts_valid):,} pts, colored by depth',
                      color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#888888', labelsize=7)
        savefig(fig, out / f'4_rect3d_{name}.png', dpi=120)

    # ════════════════════════════════════════════════════════════════
    # STEP 5: 3D renders in velodyne coords (after conversion)
    # ════════════════════════════════════════════════════════════════
    print(f'\n[Step 5] 3D renders in velodyne coords...')
    ps = subsample(pseudo[:, :3], 40000)
    d_norm = np.clip(ps[:, 0] / 60, 0, 1)

    for name, elev, azim in [('top', 90, -90), ('iso', 25, -60), ('behind', 15, -180)]:
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a2e')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#0d1117')
        ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2],
                   c=plt.cm.turbo(d_norm), s=0.1, alpha=0.6, rasterized=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X (forward)', color='white', fontsize=9)
        ax.set_ylabel('Y (left)', color='white', fontsize=9)
        ax.set_zlabel('Z (up)', color='white', fontsize=9)
        ax.set_title(f'Velodyne Coords — {name} view\n{len(pseudo):,} pts, x=[{pseudo[:,0].min():.0f},{pseudo[:,0].max():.0f}]',
                      color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#888888', labelsize=7)
        savefig(fig, out / f'5_velo3d_{name}.png', dpi=120)

    # ════════════════════════════════════════════════════════════════
    # STEP 6: BEV comparison + density analysis
    # ════════════════════════════════════════════════════════════════
    print(f'\n[Step 6] BEV comparison + domain gap analysis...')
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor('#1a1a2e')
    R = 75

    for col, (pts, label, color) in enumerate([
        (real, f'Real LiDAR — {len(real):,} pts (360°)', '#4fc3f7'),
        (pseudo, f'Pseudo-LiDAR — {len(pseudo):,} pts (~90° FOV)', '#ff8a65'),
    ]):
        ax = axes[col]
        ax.set_facecolor('#0d1117')
        p = subsample(pts, 80000)
        ax.scatter(p[:, 1], p[:, 0], c=color, s=0.1, alpha=0.5, edgecolors='none', rasterized=True)
        draw_gt_bev(ax, gt_boxes)
        ego = plt.Rectangle((-0.9, -2), 1.8, 4, lw=1.5, edgecolor='white', facecolor='#333', zorder=10)
        ax.add_patch(ego)
        ax.set_xlim(-R * 0.6, R * 0.6); ax.set_ylim(-5, R); ax.set_aspect('equal')
        ax.set_xlabel('Y (left/right) [m]', color='white')
        ax.set_ylabel('X (forward) [m]', color='white')
        ax.set_title(label, color='white', fontsize=13, fontweight='bold')
        ax.tick_params(colors='#888888')
        ax.grid(True, alpha=0.15, color='white')
        ax.invert_xaxis()
        for s in ax.spines.values(): s.set_color('#333')

    plt.tight_layout()
    savefig(fig, out / '6_bev_comparison.png')

    # ── Domain gap analysis ──
    print(f'\n  === DOMAIN GAP ANALYSIS ===')
    print(f'\n  Points in PointPillars range [0,69.12] x [-39.68,39.68] x [-3,1]:')
    def pp_range(p):
        return ((p[:, 0] >= 0) & (p[:, 0] <= 69.12) & (p[:, 1] >= -39.68) &
                (p[:, 1] <= 39.68) & (p[:, 2] >= -3) & (p[:, 2] <= 1)).sum()
    print(f'    Real:   {pp_range(real):,}')
    print(f'    Pseudo: {pp_range(pseudo):,}')

    print(f'\n  Points inside GT boxes:')
    print(f'  {"Box":>20s} {"Dist":>6s} {"Real":>8s} {"Pseudo":>8s} {"Ratio":>8s}')
    for i, b in enumerate(gt_boxes):
        rc = count_points_in_box(real, b)
        pc = count_points_in_box(pseudo, b)
        ratio = f'{pc / rc:.2f}x' if rc > 0 else 'N/A'
        print(f'  {b["cls"]+" #"+str(i):>20s} {b["dist"]:>5.0f}m {rc:>8,d} {pc:>8,d} {ratio:>8s}')

    real_fwd = real[real[:, 0] > 0]
    print(f'\n  Point density by depth (forward only):')
    print(f'  {"Band":>10s} {"Real":>8s} {"Pseudo":>8s} {"Ratio":>8s}')
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 70)]:
        r = ((real_fwd[:, 0] >= lo) & (real_fwd[:, 0] < hi)).sum()
        p = ((pseudo[:, 0] >= lo) & (pseudo[:, 0] < hi)).sum()
        ratio = f'{p / r:.2f}x' if r > 0 else 'N/A'
        print(f'  {lo:>4d}-{hi:<4d}m {r:>8,d} {p:>8,d} {ratio:>8s}')

    print(f'\n  Intensity: Real=[{real[:,3].min():.2f},{real[:,3].max():.2f}], '
          f'Pseudo=[{pseudo[:,3].min():.2f},{pseudo[:,3].max():.2f}] (constant!)')

    # Z distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('#1a1a2e')
    for col, (pts, label, color) in enumerate([
        (real_fwd, f'Real LiDAR Z distribution', '#4fc3f7'),
        (pseudo, f'Pseudo-LiDAR Z distribution', '#ff8a65'),
    ]):
        ax = axes[col]
        ax.set_facecolor('#0d1117')
        ax.hist(pts[:, 2], bins=100, range=(-4, 3), color=color, alpha=0.8, edgecolor='none')
        ax.axvline(x=-3, color='red', linestyle='--', linewidth=1, label='PP z_min=-3')
        ax.axvline(x=1, color='red', linestyle='--', linewidth=1, label='PP z_max=1')
        ax.set_xlabel('Z (up) [m]', color='white')
        ax.set_ylabel('Count', color='white')
        ax.set_title(label, color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#888888')
        ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
        for s in ax.spines.values(): s.set_color('#333')
    plt.tight_layout()
    savefig(fig, out / '6b_z_distribution.png')

    # Intensity distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0d1117')
    ax.hist(real[:, 3], bins=50, range=(0, 1), color='#4fc3f7', alpha=0.7, label=f'Real LiDAR')
    ax.axvline(x=1.0, color='#ff8a65', linewidth=3, label=f'Pseudo (all = 1.0)')
    ax.set_xlabel('Intensity', color='white')
    ax.set_ylabel('Count', color='white')
    ax.set_title('Intensity Distribution: Real vs Pseudo', color='white', fontweight='bold')
    ax.tick_params(colors='#888888')
    ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    for s in ax.spines.values(): s.set_color('#333')
    plt.tight_layout()
    savefig(fig, out / '6c_intensity.png')

    # ════════════════════════════════════════════════════════════════
    # STEP 7: PointPillars inference (optional)
    # ════════════════════════════════════════════════════════════════
    if not args.skip_inference:
        print(f'\n[Step 7] PointPillars inference...')

        print(f'  Real LiDAR...')
        rb, rs, rl = run_pp(KITTI_ROOT / f'velodyne/{idx}.bin', args.score_thresh)
        print(f'  → {len(rb)} detections')

        print(f'  Pseudo-LiDAR...')
        pb, ps_scores, pl = run_pp(KITTI_ROOT / f'pseudo-lidar_velodyne/{idx}.bin', args.score_thresh)
        print(f'  → {len(pb)} detections')

        CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']

        def draw_dets(img_base, boxes, scores, labels, color):
            out_img = img_base.copy()
            for i in range(len(boxes)):
                x, y, z, dx, dy, dz, hd = boxes[i]
                template = np.array([[1,1,-1],[1,-1,-1],[-1,-1,-1],[-1,1,-1],
                                      [1,1,1],[1,-1,1],[-1,-1,1],[-1,1,1]]) / 2.0
                corners = template * np.array([dx, dy, dz])
                c_h, s_h = np.cos(hd), np.sin(hd)
                rot = np.array([[c_h, -s_h, 0], [s_h, c_h, 0], [0, 0, 1]])
                corners = corners @ rot.T + np.array([x, y, z])
                cr = calib.project_velo_to_rect(corners)
                p2d = calib.project_rect_to_image(cr).astype(np.int32)
                hi, wi = out_img.shape[:2]
                if np.all(p2d[:, 0] < 0) or np.all(p2d[:, 0] > wi): continue
                if np.all(p2d[:, 1] < 0) or np.all(p2d[:, 1] > hi): continue
                for a, b_i in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
                    cv2.line(out_img, tuple(p2d[a]), tuple(p2d[b_i]), color, 2)
                my = np.argmin(p2d[:, 1])
                lbl = CLASS_NAMES[labels[i] - 1] if labels[i] <= len(CLASS_NAMES) else '?'
                cv2.putText(out_img, f'{lbl} {scores[i]:.2f}', (int(p2d[my, 0]), int(p2d[my, 1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            return out_img

        img_real_det = draw_dets(img, rb, rs, rl, (255, 128, 0))
        img_pseudo_det = draw_dets(img, pb, ps_scores, pl, (0, 100, 255))

        font = cv2.FONT_HERSHEY_SIMPLEX
        panels = []
        for panel_img, title, color in [
            (img_gt, f'Ground Truth ({len(gt_boxes)} boxes)', (0, 255, 0)),
            (img_real_det, f'Real LiDAR -> PointPillars ({len(rb)} det)', (255, 128, 0)),
            (img_pseudo_det, f'Pseudo-LiDAR -> PointPillars ({len(pb)} det)', (0, 100, 255)),
        ]:
            banner = np.zeros((40, panel_img.shape[1], 3), dtype=np.uint8)
            cv2.putText(banner, title, (10, 28), font, 0.7, color, 2)
            panels.append(np.vstack([banner, panel_img]))

        cv2.imwrite(str(out / '7_detection_comparison.png'), np.vstack(panels))
        print(f'  → 7_detection_comparison.png')

        print(f'\n  Summary:')
        print(f'    GT:     {len(gt_boxes)} boxes')
        print(f'    Real:   {len(rb)} detections')
        print(f'    Pseudo: {len(pb)} detections ← domain gap')
    else:
        print(f'\n[Step 7] Skipped (--skip_inference)')

    # ════════════════════════════════════════════════════════════════
    # Done
    # ════════════════════════════════════════════════════════════════
    print(f'\n{"="*70}')
    print(f'  All outputs: {out}/')
    print(f'{"="*70}\n')
    for f in sorted(out.iterdir()):
        if f.name.startswith('_'): continue
        print(f'  {f.name:45s} ({f.stat().st_size/1024:.0f} KB)')
    print(f'\n  scp -r -i ~/.ssh/kitti-project.pem ubuntu@<IP>:{out} .')


if __name__ == '__main__':
    main()