#!/usr/bin/env python3
"""
Single-Image Prototype: GroundingDINO + SAM + MapAnything → 3D Bounding Boxes
==============================================================================

Pipeline:
  1. GroundingDINO detects 2D boxes with class labels ("car", "pedestrian", "cyclist")
  2. SAM refines each box into a pixel-precise instance mask
  3. MapAnything produces per-pixel 3D points (pts3d) for the full image
  4. For each mask, gather the corresponding 3D points → fit a 3D bounding box
  5. Visualize: overlay on image + bird's-eye view comparison with KITTI GT

Run on EC2:
  source /opt/pytorch/bin/activate
  python run_single_image.py --idx 000001
"""

import argparse
import os
import sys
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# ============================================================
# PATHS — adjust if needed
# ============================================================
KITTI_ROOT = os.path.expanduser("~/kitti-project/data/KITTI/object/training")
OUTPUT_DIR = os.path.expanduser("~/sam_3d_pipeline/output")

# GroundingDINO config/weights — will be downloaded by setup script
GDINO_CONFIG = os.path.expanduser("~/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GDINO_WEIGHTS = os.path.expanduser("~/weights/groundingdino_swint_ogc.pth")
SAM_WEIGHTS = os.path.expanduser("~/weights/sam_vit_h_4b8939.pth")

# Detection prompt
TEXT_PROMPT = "car . pedestrian . person . cyclist . van . truck"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

# Map GroundingDINO labels → KITTI classes
CLASS_MAP = {
    "car": "Car", "truck": "Car", "van": "Car",
    "pedestrian": "Pedestrian", "person": "Pedestrian",
    "cyclist": "Cyclist",
}


# ============================================================
# STEP 1: GroundingDINO + SAM
# ============================================================
def run_grounding_dino_sam(image_path, device="cuda"):
    """
    Returns list of dicts:
      - mask: (H, W) bool array
      - bbox: [x1, y1, x2, y2]
      - class: KITTI class string
      - confidence: float
    """
    print("\n--- Step 1: GroundingDINO + SAM ---")

    # GroundingDINO
    sys.path.insert(0, os.path.expanduser("~/GroundingDINO"))
    from groundingdino.util.inference import load_model, load_image, predict
    from groundingdino.util import box_ops

    # SAM
    from segment_anything import sam_model_registry, SamPredictor

    # Load models
    print("  Loading GroundingDINO...")
    dino = load_model(GDINO_CONFIG, GDINO_WEIGHTS, device=device)

    print("  Loading SAM ViT-H...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_WEIGHTS)
    sam.to(device)
    predictor = SamPredictor(sam)

    # Run GroundingDINO
    image_source, image_tensor = load_image(image_path)
    boxes, logits, phrases = predict(
        model=dino, image=image_tensor, caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, device=device,
    )
    print(f"  GroundingDINO found {len(boxes)} objects: {phrases}")

    if len(boxes) == 0:
        return [], image_source

    # Convert boxes: normalized cxcywh → pixel xyxy
    h, w = image_source.shape[:2]
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h])
    boxes_np = boxes_xyxy.cpu().numpy()

    # Run SAM on each box
    image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    results = []
    for i, (box, logit, phrase) in enumerate(zip(boxes_np, logits, phrases)):
        # Map to KITTI class
        phrase_lower = phrase.lower().strip()
        kitti_class = None
        for key, val in CLASS_MAP.items():
            if key in phrase_lower:
                kitti_class = val
                break
        if kitti_class is None:
            print(f"  Skipping unknown class: '{phrase}'")
            continue

        # SAM mask from box prompt
        masks, scores, _ = predictor.predict(
            box=box[np.newaxis, :], multimask_output=False
        )

        mask = masks[0]  # (H, W) bool
        n_pixels = mask.sum()
        print(f"  [{i}] {kitti_class} (conf={logit:.2f}): {n_pixels} mask pixels")

        results.append({
            "mask": mask,
            "bbox": box.tolist(),
            "class": kitti_class,
            "confidence": float(logit),
        })

    return results, image_source


# ============================================================
# STEP 2: MapAnything depth
# ============================================================
def run_mapanything(image_path, device="cuda"):
    """
    Returns pts3d (H, W, 3) in camera coordinates and valid mask.
    """
    print("\n--- Step 2: MapAnything ---")

    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    model = MapAnything.from_pretrained("facebook/map-anything-apache").to(device)
    model.eval()

    views = load_images([image_path])
    with torch.no_grad():
        pred = model.infer(
            views, use_amp=True, amp_dtype="bf16",
            apply_mask=False, mask_edges=False, apply_confidence_mask=False
        )[0]

    pts3d = pred["pts3d"][0].cpu().numpy()       # (H, W, 3)
    depth = pred["depth_z"][0, :, :, 0].cpu().numpy()  # (H, W)

    print(f"  pts3d shape: {pts3d.shape}")
    print(f"  X range: [{pts3d[:,:,0].min():.1f}, {pts3d[:,:,0].max():.1f}]")
    print(f"  Y range: [{pts3d[:,:,1].min():.1f}, {pts3d[:,:,1].max():.1f}]")
    print(f"  Z range: [{pts3d[:,:,2].min():.1f}, {pts3d[:,:,2].max():.1f}]")

    return pts3d, depth


# ============================================================
# STEP 3: Mask → 3D points → fit 3D bbox
# ============================================================
def extract_3d_points(mask, pts3d):
    """
    Given a boolean mask and the pts3d array, return the 3D points
    belonging to that object.

    NOTE: SAM mask may be a different resolution than pts3d (MapAnything
    resizes internally). We handle that by resizing the mask.
    """
    mask_h, mask_w = mask.shape
    pts_h, pts_w = pts3d.shape[:2]

    if mask_h != pts_h or mask_w != pts_w:
        # Resize mask to match pts3d resolution
        mask_resized = cv2.resize(
            mask.astype(np.uint8), (pts_w, pts_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    else:
        mask_resized = mask

    # Extract 3D points where mask is True
    points = pts3d[mask_resized]  # (N, 3)

    # Filter invalid points (depth <= 0 or too far)
    valid = (points[:, 2] > 0.5) & (points[:, 2] < 80.0)
    points = points[valid]

    return points


def fit_3d_bbox(points, class_name="Car"):
    """
    Fit an oriented 3D bounding box around a set of 3D points.

    MapAnything pts3d is in camera coordinates:
      X = right, Y = down, Z = forward

    KITTI 3D box format:
      location (x, y, z) = bottom-center of box in camera coords
      dimensions (h, w, l) = height, width, length
      rotation_y = yaw around Y axis
    """
    if len(points) < 15:
        return None

    # --- Outlier removal (IQR on depth Z) ---
    z = points[:, 2]
    q1, q3 = np.percentile(z, [10, 90])
    iqr = q3 - q1
    keep = (z > q1 - 1.5 * iqr) & (z < q3 + 1.5 * iqr)
    points = points[keep]

    if len(points) < 15:
        return None

    # --- Yaw estimation via PCA on X-Z (bird's eye) ---
    xz = points[:, [0, 2]]
    xz_c = xz - xz.mean(axis=0)
    try:
        cov = np.cov(xz_c.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, np.argmax(eigvals)]
        ry = -np.arctan2(principal[0], principal[1])
    except:
        ry = 0.0

    # --- Rotate points to align with principal axes ---
    c, s = np.cos(ry), np.sin(ry)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    pts_rot = (R @ points.T).T

    # --- Axis-aligned bbox in rotated frame ---
    mins = pts_rot.min(axis=0)
    maxs = pts_rot.max(axis=0)

    l = maxs[2] - mins[2]  # length (Z, forward)
    w = maxs[0] - mins[0]  # width (X, lateral)
    h = maxs[1] - mins[1]  # height (Y, vertical)

    # --- Class-based dimension priors (soft clamp) ---
    priors = {
        "Car":        {"h": (1.2, 2.1), "w": (1.4, 2.2), "l": (3.0, 5.5)},
        "Pedestrian": {"h": (1.2, 2.0), "w": (0.3, 1.0), "l": (0.3, 1.0)},
        "Cyclist":    {"h": (1.2, 2.0), "w": (0.4, 1.2), "l": (1.0, 2.5)},
    }
    if class_name in priors:
        p = priors[class_name]
        h = np.clip(h, *p["h"])
        w = np.clip(w, *p["w"])
        l = np.clip(l, *p["l"])

    # --- Center: bottom-center in camera coords ---
    center_rot = np.array([
        (mins[0] + maxs[0]) / 2,
        maxs[1],                   # bottom = max Y (Y points down)
        (mins[2] + maxs[2]) / 2,
    ])
    center = R.T @ center_rot

    return {
        "x": float(center[0]), "y": float(center[1]), "z": float(center[2]),
        "h": float(h), "w": float(w), "l": float(l), "ry": float(ry),
    }


# ============================================================
# STEP 4: Visualization
# ============================================================
def visualize_results(image, detections, boxes_3d, pts3d, image_idx, output_dir):
    """Create a 3-panel figure: image+masks, depth+boxes, BEV comparison."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    colors_map = {"Car": (0, 0, 255), "Pedestrian": (255, 0, 0), "Cyclist": (0, 255, 0)}
    colors_plt = {"Car": "blue", "Pedestrian": "red", "Cyclist": "green"}

    # --- Panel 1: Image with SAM masks overlaid ---
    img_overlay = image.copy()
    for det in detections:
        mask = det["mask"]
        # Resize mask if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        color = colors_map.get(det["class"], (255, 255, 0))
        overlay = np.zeros_like(img_overlay)
        overlay[mask] = color
        img_overlay = cv2.addWeighted(img_overlay, 1.0, overlay, 0.35, 0)
        # Draw 2D bbox
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(img_overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_overlay, f"{det['class']} {det['confidence']:.2f}",
                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    axes[0].imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title("GroundingDINO + SAM detections")
    axes[0].axis("off")

    # --- Panel 2: Depth map ---
    depth_vis = pts3d[:, :, 2]  # Z = depth
    axes[1].imshow(depth_vis, cmap='turbo', vmin=0, vmax=60)
    axes[1].set_title("MapAnything depth (Z, meters)")
    axes[1].axis("off")

    # --- Panel 3: Bird's eye view (X vs Z) ---
    ax = axes[2]

    # Load GT for comparison
    gt_path = os.path.join(KITTI_ROOT, "label_2", f"{image_idx}.txt")
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            for line in f:
                p = line.strip().split()
                cls = p[0]
                if cls not in ("Car", "Pedestrian", "Cyclist"):
                    continue
                h, w, l = float(p[8]), float(p[9]), float(p[10])
                gx, gy, gz = float(p[11]), float(p[12]), float(p[13])
                gry = float(p[14])
                corners = _bev_corners(gx, gz, l, w, gry)
                poly = plt.Polygon(corners, fill=False, linestyle='--', linewidth=1.5,
                                   edgecolor=colors_plt.get(cls, 'gray'))
                ax.add_patch(poly)
                ax.text(gx, gz, f"GT {cls}", fontsize=6, ha='center', color='gray')

    # Plot predictions
    for b in boxes_3d:
        corners = _bev_corners(b['x'], b['z'], b['l'], b['w'], b['ry'])
        poly = plt.Polygon(corners, fill=False, linestyle='-', linewidth=2,
                           edgecolor=colors_plt.get(b['class'], 'orange'))
        ax.add_patch(poly)
        ax.text(b['x'], b['z'], f"{b['class']}\n{b['confidence']:.2f}",
                fontsize=7, ha='center', fontweight='bold',
                color=colors_plt.get(b['class'], 'orange'))

    ax.set_xlim(-25, 25)
    ax.set_ylim(0, 60)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m, forward)")
    ax.set_title("Bird's Eye View (dashed=GT, solid=pred)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"SAM+MapAnything → 3D Detection — Image {image_idx}", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"result_{image_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Visualization saved to {save_path}")


def _bev_corners(x, z, l, w, ry):
    c, s = np.cos(ry), np.sin(ry)
    dx = np.array([l/2, -l/2, -l/2, l/2])
    dz = np.array([w/2, w/2, -w/2, -w/2])
    cx = x + dx * c - dz * s
    cz = z + dx * s + dz * c
    return list(zip(cx, cz))


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=str, default="000001", help="KITTI image index (e.g., 000001)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--oracle", action="store_true", help="Use GT 2D boxes instead of GroundingDINO+SAM")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_path = os.path.join(KITTI_ROOT, "image_2", f"{args.idx}.png")
    assert os.path.exists(image_path), f"Image not found: {image_path}"

    image = cv2.imread(image_path)
    print(f"Image: {image_path} ({image.shape})")

    # --- Step 1: Get 2D detections + masks ---
    if args.oracle:
        print("\nUsing KITTI GT boxes as oracle...")
        h, w = image.shape[:2]
        detections = []
        gt_path = os.path.join(KITTI_ROOT, "label_2", f"{args.idx}.txt")
        with open(gt_path) as f:
            for line in f:
                p = line.strip().split()
                cls = p[0]
                if cls not in ("Car", "Pedestrian", "Cyclist"):
                    continue
                x1, y1, x2, y2 = float(p[4]), float(p[5]), float(p[6]), float(p[7])
                mask = np.zeros((h, w), dtype=bool)
                mask[int(y1):int(y2), int(x1):int(x2)] = True
                detections.append({"mask": mask, "bbox": [x1,y1,x2,y2],
                                   "class": cls, "confidence": 1.0})
        print(f"  {len(detections)} GT objects loaded")
    else:
        detections, _ = run_grounding_dino_sam(image_path, device=args.device)

    if not detections:
        print("No detections. Exiting.")
        return

    # --- Step 2: MapAnything depth ---
    pts3d, depth = run_mapanything(image_path, device=args.device)

    # --- Step 3: For each mask → extract 3D points → fit bbox ---
    print("\n--- Step 3: Fitting 3D boxes ---")
    boxes_3d = []
    for i, det in enumerate(detections):
        points = extract_3d_points(det["mask"], pts3d)
        print(f"  [{i}] {det['class']}: {len(points)} 3D points", end="")

        if len(points) < 15:
            print(" → too few, skipping")
            continue

        print(f", depth=[{points[:,2].min():.1f}, {points[:,2].max():.1f}]m", end="")

        bbox = fit_3d_bbox(points, class_name=det["class"])
        if bbox is None:
            print(" → fit failed")
            continue

        bbox["class"] = det["class"]
        bbox["confidence"] = det["confidence"]
        boxes_3d.append(bbox)
        print(f" → box at ({bbox['x']:.1f}, {bbox['y']:.1f}, {bbox['z']:.1f}), "
              f"dims=({bbox['h']:.1f}, {bbox['w']:.1f}, {bbox['l']:.1f})")

    print(f"\n  Total 3D boxes: {len(boxes_3d)}")

    # --- Step 4: Save KITTI-format predictions ---
    pred_path = os.path.join(args.output_dir, f"{args.idx}.txt")
    with open(pred_path, 'w') as f:
        for b in boxes_3d:
            f.write(f"{b['class']} 0.0 0 0.0 0 0 0 0 "
                    f"{b['h']:.2f} {b['w']:.2f} {b['l']:.2f} "
                    f"{b['x']:.2f} {b['y']:.2f} {b['z']:.2f} "
                    f"{b['ry']:.2f} {b['confidence']:.4f}\n")
    print(f"  Predictions saved to {pred_path}")

    # --- Step 5: Visualize ---
    visualize_results(image, detections, boxes_3d, pts3d, args.idx, args.output_dir)

    # --- Print GT comparison ---
    print("\n--- Ground Truth Comparison ---")
    gt_path = os.path.join(KITTI_ROOT, "label_2", f"{args.idx}.txt")
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            for line in f:
                p = line.strip().split()
                if p[0] in ("Car", "Pedestrian", "Cyclist"):
                    print(f"  GT: {p[0]:12s} at ({float(p[11]):6.1f}, {float(p[12]):6.1f}, {float(p[13]):6.1f}), "
                          f"dims=({float(p[8]):.1f}, {float(p[9]):.1f}, {float(p[10]):.1f})")
    for b in boxes_3d:
        print(f"  PR: {b['class']:12s} at ({b['x']:6.1f}, {b['y']:6.1f}, {b['z']:6.1f}), "
              f"dims=({b['h']:.1f}, {b['w']:.1f}, {b['l']:.1f})")


if __name__ == "__main__":
    main()
