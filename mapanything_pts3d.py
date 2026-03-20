import os
import torch
import numpy as np
from pathlib import Path
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def run_inference(image_dir, output_dir, split_file=None):
    print("=== Loading MapAnything model ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = MapAnything.from_pretrained("facebook/map-anything-apache").to(device)
    model.eval()
    print("Model loaded.")

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if split_file:
        with open(split_file) as f:
            stems = [line.strip() for line in f if line.strip()]
        image_paths = [image_dir / f"{s}.png" for s in stems]
    else:
        image_paths = sorted(image_dir.glob("*.png"))

    print(f"Found {len(image_paths)} images to process")
    # Resume: skip if both pts3d and mask already saved
    image_paths = [p for p in image_paths if not (output_dir / (p.stem + "_pts3d.npy")).exists()]
    print(f"{len(image_paths)} remaining after resume check")

    total = len(image_paths)
    processed = 0
    failed = 0
    start_time = time.time()

    for img_path in image_paths:
        try:
            views = load_images([str(img_path)])
            with torch.no_grad():
                predictions = model.infer(
                    views,
                    use_amp=False,
                    apply_mask=False,
                    mask_edges=False,
                    apply_confidence_mask=False,
                )

            pred = predictions[0]

            # pts3d: (1, H, W, 3) -> (H, W, 3)
            pts3d = pred["pts3d"][0].cpu().numpy().astype(np.float32)
            # non_ambiguous_mask is the valid point mask
            mask = pred["non_ambiguous_mask"][0].cpu().numpy().astype(bool).squeeze()

            np.save(str(output_dir / (img_path.stem + "_pts3d.npy")), pts3d)
            np.save(str(output_dir / (img_path.stem + "_mask.npy")), mask)

            processed += 1

        except Exception as e:
            import traceback
            print(f"FAIL {img_path.stem}: {e}")
            traceback.print_exc()
            failed += 1

        if processed % 50 == 0 and processed > 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            remaining = (total - processed) / rate if rate > 0 else 0
            print(f"Progress: {processed}/{total} | {failed} failed | Rate: {rate:.2f} img/s | ETA: {remaining/60:.1f} min")

    print(f"=== Done. {processed} success, {failed} failed out of {total} ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_file", default=None)
    args = parser.parse_args()
    run_inference(args.image_dir, args.output_dir, args.split_file)