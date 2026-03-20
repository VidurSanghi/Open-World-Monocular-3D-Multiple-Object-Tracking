#!/usr/bin/env python3
"""
3-Way Comparison: Ground Truth vs Real LiDAR PointPillars vs Pseudo-LiDAR PointPillars

Runs each inference in a separate subprocess to avoid OpenPCDet global cfg contamination.

Usage:
    source /opt/pytorch/bin/activate
    python compare_3way.py --idx 000008
    python compare_3way.py --idx 000008 --score_thresh 0.3

Output: ~/kitti-project/output/compare_{idx}.png
"""

import sys
import os
import json
import argparse
import subprocess
import tempfile
import numpy as np
import cv2
from pathlib import Path

# ── Paths ──
KITTI_ROOT = Path('/home/ubuntu/kitti-project/data/KITTI/object/training')
OUTPUT_DIR = Path('/home/ubuntu/kitti-project/output')

# Colors (BGR for OpenCV)
COLOR_GT     = (0, 255, 0)    # green
COLOR_REAL   = (255, 128, 0)  # cyan-ish in BGR
COLOR_PSEUDO = (0, 100, 255)  # orange in BGR

CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']


# ═══════════════════════════════════════════════════════════════════
# Calibration
# ═══════════════════════════════════════════════════════════════════

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

    def cart_to_hom(self, pts):
        return np.hstack([pts, np.ones((pts.shape[0], 1))])

    def project_velo_to_image(self, pts_3d_velo):
        pts_hom = self.cart_to_hom(pts_3d_velo)
        pts_cam = (self.R0 @ self.V2C @ pts_hom.T)
        pts_2d = self.P2 @ pts_cam
        pts_2d[0, :] /= pts_2d[2, :]
        pts_2d[1, :] /= pts_2d[2, :]
        return pts_2d[:2, :].T

    def project_velo_to_rect(self, pts_3d_velo):
        pts_hom = self.cart_to_hom(pts_3d_velo)
        pts_rect = (self.R0 @ self.V2C @ pts_hom.T).T
        return pts_rect[:, :3]


# ═══════════════════════════════════════════════════════════════════
# 3D Box utilities
# ═══════════════════════════════════════════════════════════════════

def boxes_to_corners_3d(boxes):
    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1],  [1, -1, 1],  [-1, -1, 1],  [-1, 1, 1]
    ]) / 2.0

    corners_all = []
    for box in boxes:
        x, y, z, dx, dy, dz, heading = box
        corners = template * np.array([dx, dy, dz])
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rot = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])
        corners = corners @ rot.T + np.array([x, y, z])
        corners_all.append(corners)
    return np.array(corners_all)


def gt_label_to_lidar_box(label_line, calib):
    parts = label_line.strip().split()
    cls = parts[0]
    if cls == 'DontCare':
        return None, None

    h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
    x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
    ry = float(parts[14])

    center_rect = np.array([[x, y - h / 2, z]])
    R0_inv = np.linalg.inv(calib.R0)
    V2C_inv = np.linalg.inv(calib.V2C)
    pts_hom = np.hstack([center_rect, np.ones((1, 1))])
    center_velo = (V2C_inv @ R0_inv @ pts_hom.T).T[0, :3]

    heading_velo = -(ry + np.pi / 2)
    box_velo = np.array([center_velo[0], center_velo[1], center_velo[2], l, w, h, heading_velo])
    return box_velo, cls


def draw_box_3d_on_image(img, corners_velo, calib, color, thickness=2):
    pts_2d = calib.project_velo_to_image(corners_velo).astype(np.int32)
    h, w = img.shape[:2]
    if np.all(pts_2d[:, 0] < 0) or np.all(pts_2d[:, 0] > w):
        return
    if np.all(pts_2d[:, 1] < 0) or np.all(pts_2d[:, 1] > h):
        return
    pts_cam = calib.project_velo_to_rect(corners_velo)
    if np.all(pts_cam[:, 2] < 0.1):
        return

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness)


# ═══════════════════════════════════════════════════════════════════
# Subprocess inference — each run gets a fresh Python process
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

class SingleFile(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, bin_path, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=False, root_path=None, logger=logger)
        self.bin_path = bin_path
    def __len__(self):
        return 1
    def __getitem__(self, index):
        points = np.fromfile(self.bin_path, dtype=np.float32).reshape(-1, 4)
        return self.prepare_data({'points': points, 'frame_id': 0})

bin_path = sys.argv[1]
score_thresh = float(sys.argv[2])
output_path = sys.argv[3]

logger = common_utils.create_logger()
cfg_from_yaml_file('/home/ubuntu/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml', cfg)

ds = SingleFile(cfg.DATA_CONFIG, cfg.CLASS_NAMES, bin_path, logger)
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
model.load_params_from_file(filename='/home/ubuntu/OpenPCDet/pointpillar_7728.pth',
                            logger=logger, to_cpu=True)
model.cuda()
model.eval()

with torch.no_grad():
    d = ds[0]
    d = ds.collate_batch([d])
    load_data_to_gpu(d)
    pred, _ = model.forward(d)

boxes = pred[0]['pred_boxes'].cpu().numpy()
scores = pred[0]['pred_scores'].cpu().numpy()
labels = pred[0]['pred_labels'].cpu().numpy()

mask = scores >= score_thresh
result = {
    'boxes': boxes[mask].tolist(),
    'scores': scores[mask].tolist(),
    'labels': labels[mask].tolist(),
}
with open(output_path, 'w') as f:
    json.dump(result, f)
'''


def run_pointpillars_subprocess(bin_path, score_thresh):
    """Run PointPillars in a clean subprocess. Returns (boxes, scores, labels)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_f:
        script_f.write(INFERENCE_SCRIPT)
        script_path = script_f.name

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as out_f:
        output_path = out_f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path, str(bin_path), str(score_thresh), output_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f'  STDERR (last 500 chars): {result.stderr[-500:]}')
            raise RuntimeError(f'Inference subprocess failed (exit {result.returncode})')

        with open(output_path) as f:
            data = json.load(f)

        boxes = np.array(data['boxes']).reshape(-1, 7) if data['boxes'] else np.zeros((0, 7))
        scores = np.array(data['scores']) if data['scores'] else np.zeros(0)
        labels = np.array(data['labels'], dtype=np.int32) if data['labels'] else np.zeros(0, dtype=np.int32)
        return boxes, scores, labels
    finally:
        os.unlink(script_path)
        os.unlink(output_path)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=str, default='000008')
    parser.add_argument('--score_thresh', type=float, default=0.3)
    args = parser.parse_args()

    idx = args.idx
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  3-Way Comparison for KITTI sample {idx}')
    print(f'{"="*60}\n')

    img_path = KITTI_ROOT / f'image_2/{idx}.png'
    calib_path = KITTI_ROOT / f'calib/{idx}.txt'
    label_path = KITTI_ROOT / f'label_2/{idx}.txt'
    real_lidar_path = KITTI_ROOT / f'velodyne/{idx}.bin'
    pseudo_lidar_path = KITTI_ROOT / f'pseudo-lidar_velodyne/{idx}.bin'

    for p, name in [(img_path, 'Image'), (calib_path, 'Calib'), (label_path, 'Labels'),
                     (real_lidar_path, 'Real LiDAR'), (pseudo_lidar_path, 'Pseudo-LiDAR')]:
        if not p.exists():
            print(f'ERROR: {name} not found: {p}')
            sys.exit(1)
        print(f'  ✓ {name}: {p.name}')

    img_base = cv2.imread(str(img_path))
    calib = Calibration(calib_path)

    # ── Panel 1: Ground Truth ──
    print(f'\n[1/3] Drawing ground truth boxes...')
    img_gt = img_base.copy()
    gt_count = 0
    with open(label_path) as f:
        for line in f:
            box_velo, cls = gt_label_to_lidar_box(line, calib)
            if box_velo is None:
                continue
            corners = boxes_to_corners_3d(box_velo.reshape(1, 7))[0]
            draw_box_3d_on_image(img_gt, corners, calib, COLOR_GT, thickness=2)
            gt_count += 1
    print(f'  → {gt_count} ground truth boxes drawn')

    # ── Panel 2: Real LiDAR (isolated subprocess) ──
    print(f'\n[2/3] Running PointPillars on real LiDAR (subprocess)...')
    real_pts = np.fromfile(str(real_lidar_path), dtype=np.float32).reshape(-1, 4)
    print(f'  → Real LiDAR: {real_pts.shape[0]:,} points')
    real_boxes, real_scores, real_labels = run_pointpillars_subprocess(
        real_lidar_path, args.score_thresh
    )
    print(f'  → {len(real_boxes)} detections (score >= {args.score_thresh})')
    if len(real_scores) > 0:
        print(f'  → Scores: {np.round(real_scores, 2)}')

    img_real = img_base.copy()
    for i in range(len(real_boxes)):
        corners = boxes_to_corners_3d(real_boxes[i:i+1])[0]
        draw_box_3d_on_image(img_real, corners, calib, COLOR_REAL, thickness=2)
        pts_2d = calib.project_velo_to_image(corners)
        min_y_idx = np.argmin(pts_2d[:, 1])
        tx, ty = int(pts_2d[min_y_idx, 0]), int(pts_2d[min_y_idx, 1]) - 5
        label_name = CLASS_NAMES[real_labels[i] - 1] if real_labels[i] <= len(CLASS_NAMES) else '?'
        cv2.putText(img_real, f'{label_name} {real_scores[i]:.2f}',
                     (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_REAL, 1)

    # ── Panel 3: Pseudo-LiDAR (isolated subprocess) ──
    print(f'\n[3/3] Running PointPillars on pseudo-LiDAR (subprocess)...')
    pseudo_pts = np.fromfile(str(pseudo_lidar_path), dtype=np.float32).reshape(-1, 4)
    print(f'  → Pseudo-LiDAR: {pseudo_pts.shape[0]:,} points')
    pseudo_boxes, pseudo_scores, pseudo_labels = run_pointpillars_subprocess(
        pseudo_lidar_path, args.score_thresh
    )
    print(f'  → {len(pseudo_boxes)} detections (score >= {args.score_thresh})')
    if len(pseudo_scores) > 0:
        print(f'  → Scores: {np.round(pseudo_scores, 2)}')

    img_pseudo = img_base.copy()
    for i in range(len(pseudo_boxes)):
        corners = boxes_to_corners_3d(pseudo_boxes[i:i+1])[0]
        draw_box_3d_on_image(img_pseudo, corners, calib, COLOR_PSEUDO, thickness=2)
        pts_2d = calib.project_velo_to_image(corners)
        min_y_idx = np.argmin(pts_2d[:, 1])
        tx, ty = int(pts_2d[min_y_idx, 0]), int(pts_2d[min_y_idx, 1]) - 5
        label_name = CLASS_NAMES[pseudo_labels[i] - 1] if pseudo_labels[i] <= len(CLASS_NAMES) else '?'
        cv2.putText(img_pseudo, f'{label_name} {pseudo_scores[i]:.2f}',
                     (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_PSEUDO, 1)

    # ── Compose final image ──
    print(f'\nComposing output...')
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_h = 40
    panels = []
    for img_panel, title, color in [
        (img_gt,     f'Ground Truth ({gt_count} boxes)', COLOR_GT),
        (img_real,   f'Real LiDAR -> PointPillars ({len(real_boxes)} det, thresh={args.score_thresh})', COLOR_REAL),
        (img_pseudo, f'Pseudo-LiDAR -> PointPillars ({len(pseudo_boxes)} det, thresh={args.score_thresh})', COLOR_PSEUDO),
    ]:
        banner = np.zeros((label_h, img_panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, title, (10, 28), font, 0.7, color, 2)
        panel = np.vstack([banner, img_panel])
        panels.append(panel)

    output = np.vstack(panels)
    out_path = OUTPUT_DIR / f'compare_{idx}.png'
    cv2.imwrite(str(out_path), output)

    print(f'\n{"="*60}')
    print(f'  ✓ Saved: {out_path}')
    print(f'  Image size: {output.shape[1]}x{output.shape[0]}')
    print(f'{"="*60}')
    print(f'\nSummary:')
    print(f'  Ground truth:  {gt_count} boxes')
    print(f'  Real LiDAR:    {len(real_boxes)} detections')
    print(f'  Pseudo-LiDAR:  {len(pseudo_boxes)} detections  ← expected ~0 (domain gap)')
    print(f'\nTo download: scp -i ~/.ssh/kitti-project.pem ubuntu@<IP>:{out_path} .')


if __name__ == '__main__':
    main()