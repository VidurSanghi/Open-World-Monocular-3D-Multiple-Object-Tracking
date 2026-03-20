import sys
sys.path.insert(0, '/home/ubuntu/pseudo_lidar/preprocessing')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

IDX  = "000001"
DATA = '/home/ubuntu/kitti-project/data/KITTI/object/training'

img = mpimg.imread(f'{DATA}/image_2/{IDX}.png')

with open(f'{DATA}/label_2/{IDX}.txt') as f:
    labels = [l.strip().split() for l in f if l.strip()]

colors = {'Car':'lime', 'Pedestrian':'cyan', 'Cyclist':'yellow',
          'Van':'lime', 'Truck':'lime', 'Person_sitting':'cyan'}

fig, ax = plt.subplots(1,1, figsize=(12,5))
ax.imshow(img)

count = 0
for parts in labels:
    cls = parts[0]
    if cls not in colors:
        continue
    x1,y1,x2,y2 = float(parts[4]),float(parts[5]),float(parts[6]),float(parts[7])
    ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,
                 linewidth=2, edgecolor=colors[cls], facecolor='none', linestyle='--'))
    ax.text(x1, y1-4, cls, color=colors[cls], fontsize=8,
            bbox=dict(facecolor='black', alpha=0.5, pad=1))
    count += 1

ax.set_title(f'Ground Truth Labels | {count} objects | {IDX}')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'/home/ubuntu/kitti-project/ground_truth_{IDX}.png', dpi=150, bbox_inches='tight')
print(f"Saved ground_truth_{IDX}.png - {count} objects")
for p in labels:
    if p[0] in colors:
        print(f"  {p[0]} at xyz=({p[11]},{p[12]},{p[13]})")