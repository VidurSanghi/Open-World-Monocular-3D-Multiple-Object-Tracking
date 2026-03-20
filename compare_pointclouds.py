#!/usr/bin/env python3
"""
Point Cloud Comparison: Real LiDAR vs Pseudo-LiDAR (MapAnything)

Shows bird's eye view and front view of both point clouds.
Ground truth boxes shown as dashed outlines clearly labeled "GT" in legend.

Usage:
    source /opt/pytorch/bin/activate
    python compare_pointclouds.py --idx 000008

Output: ~/kitti-project/output/pointclouds_{idx}.png
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from pathlib import Path

KITTI_ROOT = Path('/home/ubuntu/kitti-project/data/KITTI/object/training')
OUTPUT_DIR = Path('/home/ubuntu/kitti-project/output')


class Calibration:
    def __init__(self, calib_path):
        calib = {}
        with open(calib_path) as f:
            for line in f:
                if ':' not in line:
                    continue
                key, val = line.split(':', 1)
                calib[key.strip()] = np.array([float(x) for x in val.split()])
        self.P2 = calib['P2'].reshape(3, 4)
        self.R0 = np.eye(4)
        self.R0[:3, :3] = calib['R0_rect'].reshape(3, 3)
        self.V2C = np.eye(4)
        self.V2C[:3, :4] = calib['Tr_velo_to_cam'].reshape(3, 4)

    def rect_to_velo(self, pts_rect):
        pts_hom = np.hstack([pts_rect, np.ones((pts_rect.shape[0], 1))])
        R0_inv = np.linalg.inv(self.R0)
        V2C_inv = np.linalg.inv(self.V2C)
        return (V2C_inv @ R0_inv @ pts_hom.T).T[:, :3]


def parse_gt_boxes(label_path, calib):
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
            center_velo = calib.rect_to_velo(center_rect)[0]
            heading = -(ry + np.pi / 2)
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            dx, dy = l / 2, w / 2
            corners_local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
            rot2d = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            corners_bev = (rot2d @ corners_local.T).T + center_velo[:2]
            x_corners = corners_bev[:, 0]
            z_lo = center_velo[2] - h / 2
            z_hi = center_velo[2] + h / 2
            corners_front = np.array([
                [x_corners.min(), z_lo], [x_corners.max(), z_lo],
                [x_corners.max(), z_hi], [x_corners.min(), z_hi],
            ])
            boxes.append({
                'cls': cls, 'corners_bev': corners_bev,
                'corners_front': corners_front, 'center_velo': center_velo,
                'depth': center_velo[0],
            })
    return boxes


def draw_gt_bev(ax, gt_boxes):
    for b in gt_boxes:
        polygon = plt.Polygon(b['corners_bev'], fill=False, edgecolor='#00ff88',
                               linewidth=1.5, linestyle='--', zorder=5)
        ax.add_patch(polygon)
        ax.text(b['center_velo'][0], b['center_velo'][1],
                f"{b['cls']}\n{b['depth']:.0f}m",
                fontsize=6, color='#00ff88', ha='center', va='center',
                zorder=6, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground='black')])


def draw_gt_front(ax, gt_boxes):
    for b in gt_boxes:
        polygon = plt.Polygon(b['corners_front'], fill=False, edgecolor='#00ff88',
                               linewidth=1.5, linestyle='--', zorder=5)
        ax.add_patch(polygon)


def add_legend(ax):
    legend_elements = [
        Line2D([0], [0], color='#00ff88', linewidth=1.5, linestyle='--', label='Ground Truth Box'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6600',
               markersize=5, linestyle='None', label='Point Cloud'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
              facecolor='#1a1a2e', edgecolor='#333333', labelcolor='white')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=str, default='000008')
    parser.add_argument('--bev_range', type=float, default=75)
    args = parser.parse_args()

    idx = args.idx
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calib = Calibration(KITTI_ROOT / f'calib/{idx}.txt')
    real_pts = np.fromfile(str(KITTI_ROOT / f'velodyne/{idx}.bin'), dtype=np.float32).reshape(-1, 4)
    pseudo_pts = np.fromfile(str(KITTI_ROOT / f'pseudo-lidar_velodyne/{idx}.bin'), dtype=np.float32).reshape(-1, 4)
    gt_boxes = parse_gt_boxes(KITTI_ROOT / f'label_2/{idx}.txt', calib)

    n_real, n_pseudo, n_gt = len(real_pts), len(pseudo_pts), len(gt_boxes)
    print(f'Sample {idx}: Real={n_real:,} pts, Pseudo={n_pseudo:,} pts, GT={n_gt} boxes')

    fig, axes = plt.subplots(2, 2, figsize=(20, 16),
                              gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor('#1a1a2e')
    R = args.bev_range
    ps = 0.15  # point size

    # ── Color scheme: real=cyan/blue, pseudo=orange/warm ──
    real_bev_color = '#4fc3f7'    # light blue
    pseudo_bev_color = '#ff8a65'  # orange

    # ════════════════════════════════════════
    # Row 1: Bird's Eye View
    # ════════════════════════════════════════
    for col, (pts, label, n, color) in enumerate([
        (real_pts, 'Real LiDAR (Velodyne HDL-64E)', n_real, real_bev_color),
        (pseudo_pts, 'Pseudo-LiDAR (MapAnything)', n_pseudo, pseudo_bev_color),
    ]):
        ax = axes[0, col]
        ax.set_facecolor('#0d1117')

        ax.scatter(pts[:, 1], pts[:, 0], c=color, s=ps,
                   alpha=0.5, edgecolors='none', rasterized=True)

        draw_gt_bev(ax, gt_boxes)

        # Ego vehicle
        ego = plt.Rectangle((-0.9, -2.0), 1.8, 4.0, linewidth=1.5,
                             edgecolor='white', facecolor='#333333', zorder=10)
        ax.add_patch(ego)
        ax.text(0, 0, '▲', fontsize=10, color='white', ha='center',
                va='center', zorder=11, fontweight='bold')

        ax.set_xlim(-R * 0.6, R * 0.6)
        ax.set_ylim(-5, R)
        ax.set_aspect('equal')
        ax.set_xlabel('Y (left ←→ right) [m]', color='white', fontsize=10)
        ax.set_ylabel('X (forward →) [m]', color='white', fontsize=10)
        ax.set_title(f"Bird's Eye View: {label}\n{n:,} points  |  {n_gt} GT boxes (dashed green)",
                      color='white', fontsize=12, fontweight='bold', pad=12)
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_color('#333333')
        ax.grid(True, alpha=0.15, color='white')
        ax.invert_xaxis()
        add_legend(ax)

    # ════════════════════════════════════════
    # Row 2: Front View (X vs Z)
    # ════════════════════════════════════════
    for col, (pts, label, n, color) in enumerate([
        (real_pts, 'Real LiDAR', n_real, real_bev_color),
        (pseudo_pts, 'Pseudo-LiDAR', n_pseudo, pseudo_bev_color),
    ]):
        ax = axes[1, col]
        ax.set_facecolor('#0d1117')

        fwd = pts[pts[:, 0] > 0]
        n_fwd = len(fwd)

        ax.scatter(fwd[:, 0], fwd[:, 2], c=color, s=ps * 0.5,
                   alpha=0.4, edgecolors='none', rasterized=True)

        draw_gt_front(ax, gt_boxes)

        ax.set_xlim(0, R)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('X (forward →) [m]', color='white', fontsize=10)
        ax.set_ylabel('Z (up ↑) [m]', color='white', fontsize=10)
        ax.set_title(f'Front View: {label}  ({n_fwd:,} forward points)',
                      color='white', fontsize=11, fontweight='bold', pad=8)
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_color('#333333')
        ax.grid(True, alpha=0.15, color='white')

    # ── Summary stats bar ──
    stats = (
        f"Sample: {idx}  |  "
        f"Real LiDAR: {n_real:,} pts (360°, Velodyne HDL-64E)  |  "
        f"Pseudo-LiDAR: {n_pseudo:,} pts (~90° FOV, MapAnything)  |  "
        f"Ratio: {n_real / n_pseudo:.1f}x  |  "
        f"GT: {n_gt} objects  |  "
        f"Green dashed = ground truth (not detections)"
    )
    fig.text(0.5, 0.01, stats, ha='center', va='bottom',
             fontsize=9, color='#aaaaaa', fontstyle='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = OUTPUT_DIR / f'pointclouds_{idx}.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'✓ Saved: {out_path}')


if __name__ == '__main__':
    main()