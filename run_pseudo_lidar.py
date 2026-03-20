import sys
import os
sys.path.insert(0, '/home/ubuntu/OpenPCDet')
sys.path.insert(0, '/home/ubuntu/pseudo_lidar/preprocessing')
os.chdir('/home/ubuntu/OpenPCDet/tools')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, box_utils
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import kitti_util

IDX   = "000001"
CKPT  = '/home/ubuntu/OpenPCDet/pointpillar_7728.pth'
CFG   = '/home/ubuntu/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml'
DATA  = '/home/ubuntu/kitti-project/data/KITTI/object/training'
SCORE = 0.3

cfg_from_yaml_file(CFG, cfg)
cfg.DATA_CONFIG.DATA_SPLIT.test = 'val'
logger = common_utils.create_logger()
dataset = KittiDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                       root_path=None, training=False, logger=logger)
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
model.load_params_from_file(CKPT, logger=logger)
model.cuda(); model.eval()

idx_int = next(i for i,info in enumerate(dataset.kitti_infos)
               if info['point_cloud']['lidar_idx'] == IDX)
data  = dataset[idx_int]
batch = dataset.collate_batch([data])
load_data_to_gpu(batch)

# --- swap in PSEUDO lidar ---
pts = np.fromfile(f'{DATA}/pseudo-lidar_velodyne/{IDX}.bin', dtype=np.float32).reshape(-1,4)
pts_b = np.hstack([np.zeros((len(pts),1), dtype=np.float32), pts])
batch['points'] = torch.from_numpy(pts_b).cuda()

with torch.no_grad():
    pred, _ = model.forward(batch)

boxes  = pred[0]['pred_boxes'].cpu().numpy()
scores = pred[0]['pred_scores'].cpu().numpy()
labels = pred[0]['pred_labels'].cpu().numpy()
keep   = scores >= SCORE
boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
print(f"Pseudo-LiDAR detections: {len(boxes)}")

calib = kitti_util.Calibration(f'{DATA}/calib/{IDX}.txt')
img   = mpimg.imread(f'{DATA}/image_2/{IDX}.png')
h, w  = img.shape[:2]

def proj_box(box):
    corners = box_utils.boxes_to_corners_3d(box[np.newaxis])[0]
    rect    = calib.project_velo_to_rect(corners)
    pts2d   = calib.project_rect_to_image(rect)
    u1,v1   = pts2d[:,0].min(), pts2d[:,1].min()
    u2,v2   = pts2d[:,0].max(), pts2d[:,1].max()
    return [max(0,u1), max(0,v1), min(w,u2), min(h,v2)]

fig, ax = plt.subplots(1,1, figsize=(12,5))
ax.imshow(img)
colors = {1:'red', 2:'orange', 3:'magenta'}
for box, score, label in zip(boxes, scores, labels):
    x1,y1,x2,y2 = proj_box(box)
    ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,
                 linewidth=2, edgecolor=colors.get(label,'white'), facecolor='none'))
    ax.text(x1, y1-4, f'{cfg.CLASS_NAMES[label-1]} {score:.2f}',
            color=colors.get(label,'white'), fontsize=8,
            bbox=dict(facecolor='black', alpha=0.5, pad=1))
ax.set_title(f'Pseudo-LiDAR (MapAnything) → PointPillars | {len(boxes)} detections | {IDX}')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'/home/ubuntu/kitti-project/pseudo_lidar_detections_{IDX}.png', dpi=150, bbox_inches='tight')
print("Saved pseudo_lidar_detections.png")