"""Load the pretrained student and report normal MAE on test_imgs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.io.multicolorps import MultiColorPS
from modules.model.student_model import StudentModel
from modules.utils.compute_mae import compute_mae


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate StudentModel mean angular error on test_imgs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "test_imgs" / "SpectraM14_Multiplexing",
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model.pt")
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=1024,
        help="Square resize cap used by MultiColorPS (multiple of 256).",
    )
    parser.add_argument("--pixel-samples", type=int, default=4096)
    parser.add_argument("--canonical-resolution", type=int, default=128)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N samples. 0 means the full set.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_student(checkpoint, device, pixel_samples, canonical_resolution):
    model = StudentModel(
        pixel_samples=pixel_samples,
        device=device,
        canonical_resolution=canonical_resolution,
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    if any(str(k).startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model_keys = set(model.state_dict())
    ckpt_keys = set(state)
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    if missing or unexpected:
        raise RuntimeError(
            "StudentModel does not match the checkpoint. "
            f"missing={len(missing)} unexpected={len(unexpected)}. "
            f"missing sample={missing[:8]} unexpected sample={unexpected[:8]}"
        )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, len(model_keys)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    dataset = MultiColorPS(
        data_root=str(args.data_root),
        type="test",
        num_images=1,
        test_img_min=1,
        test_img_max=1,
        max_image_resolution=args.max_resolution,
        augment=False,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No test samples found under {args.data_root}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model, n_tensors = load_student(
        args.checkpoint, device, args.pixel_samples, args.canonical_resolution
    )
    print(
        f"Loaded {args.checkpoint.name} into StudentModel "
        f"({n_tensors} tensors, strict). device={device.type}"
    )
    print(
        "architecture: use_light_calib_model="
        f"{model.use_light_calib_model} use_per_pixel_branch="
        f"{model.use_per_pixel_branch} pp_out_ch={model.pp_out_ch} "
        f"input_dim={model.input_dim}"
    )

    total_error = 0.0
    total_mask = 0.0
    n_done = 0
    for (class_path, _), batch in zip(dataset.samples, loader):
        if args.limit and n_done >= args.limit:
            break
        img, normal, mask, _shading, n_img, _lights = batch
        img = img.to(device, dtype=torch.float32)
        normal = normal.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.float32)
        if not torch.is_tensor(n_img):
            n_img = torch.tensor([n_img], device=device, dtype=torch.long)
        else:
            n_img = n_img.to(device=device, dtype=torch.long)

        _, _, height, width, _ = img.shape
        model.set_inference_context(
            mask=mask,
            nImgArray=n_img,
            canonical_resolution=args.canonical_resolution,
            decoder_resolution=height,
        )
        with torch.inference_mode():
            pred = model(img)
        pred = torch.nn.functional.normalize(pred, dim=1, eps=1e-6)
        gt = torch.nn.functional.normalize(normal, dim=1, eps=1e-6)
        if pred.shape[-2:] != gt.shape[-2:]:
            raise RuntimeError(
                f"Prediction spatial size {tuple(pred.shape[-2:])} != "
                f"ground truth {tuple(gt.shape[-2:])} for {class_path}"
            )
        mae, emap = compute_mae(pred, gt, mask)
        weight = float(mask.sum().item())
        total_error += float(emap.sum().item())
        total_mask += weight
        n_done += 1
        name = Path(class_path).name
        print(f"{name:12s}  MAE {float(mae):7.3f} deg   HxW {height}x{width}")

    if total_mask <= 0:
        raise RuntimeError("Mask is empty; MAE is undefined.")
    overall = total_error / total_mask
    print(f"overall MAE {overall:.3f} deg  ({n_done} objects, mask-weighted)")
    return overall


if __name__ == "__main__":
    main()
