# eval_tta.py
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

# === from your project ===
from dataset_sar_rarp50 import SarRarp50Seg, NUM_CLASSES, CLASS_NAMES
import segmentation_models_pytorch as smp

# -----------------------------
# Build model (same as train.py)
# -----------------------------
def build_model(name="unet", encoder="resnet34"):
    name = name.lower()
    if name == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,  # weights are loaded from checkpoint
            in_channels=3,
            classes=NUM_CLASSES,
        )
    elif name in ("deeplabv3plus", "deeplab"):
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Use 'unet' or 'deeplabv3plus'.")

# -----------------------------
# IoU utilities (same logic as train.py)
# -----------------------------
def init_conf(num_classes):
    inter = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    return inter, union

def update_conf_from_logits(inter, union, logits, target):
    with torch.no_grad():
        pred = logits.argmax(1)
        for c in range(NUM_CLASSES):
            p = (pred == c)
            t = (target == c)
            inter[c] += (p & t).sum().item()
            union[c] += (p | t).sum().item()

def finalize_iou(inter, union):
    per_class_iou = np.full(len(inter), np.nan, dtype=np.float64)
    for c in range(len(inter)):
        if union[c] > 0:
            per_class_iou[c] = inter[c] / union[c]
    m_iou = np.nanmean(per_class_iou) if np.any(~np.isnan(per_class_iou)) else 0.0
    return per_class_iou, float(m_iou)

# -----------------------------
# NSD (normalized surface dice) — same as in your train.py
# -----------------------------
def _binary_boundaries(mask_uint8):
    edges = cv2.Canny(mask_uint8 * 255, 100, 200)
    return (edges > 0).astype(np.uint8)

def compute_nsd_binary(pred_bin: np.ndarray, gt_bin: np.ndarray, tau: int = 3) -> float:
    p = (pred_bin.astype(np.uint8) > 0).astype(np.uint8)
    g = (gt_bin.astype(np.uint8) > 0).astype(np.uint8)

    pb = _binary_boundaries(p)
    gb = _binary_boundaries(g)

    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0

    dist_p = cv2.distanceTransform((1 - pb).astype(np.uint8), cv2.DIST_L2, 0)
    dist_g = cv2.distanceTransform((1 - gb).astype(np.uint8), cv2.DIST_L2, 0)

    match_p = (dist_g[pb > 0] <= tau).sum() if pb.sum() > 0 else 0
    match_g = (dist_p[gb > 0] <= tau).sum() if gb.sum() > 0 else 0

    total = int(pb.sum() + gb.sum())
    nsd = (match_p + match_g) / total if total > 0 else 1.0
    return float(nsd)

# -----------------------------
# TTA inference (flip + multi-scale)
# -----------------------------
def tta_forward(model, x, hflip=False, scales=None):
    outs = []
    B, C, H, W = x.shape

    # base
    outs.append(model(x))

    # flip
    if hflip:
        xf = torch.flip(x, dims=[3])
        pf = model(xf)
        pf = torch.flip(pf, dims=[3])
        outs.append(pf)

    # scales
    if scales:
        for s in scales:
            if abs(s - 1.0) < 1e-6:
                continue
            Hs, Ws = int(H * s), int(W * s)
            xs = F.interpolate(x, size=(Hs, Ws), mode="bilinear", align_corners=False)
            ps = model(xs)
            ps = F.interpolate(ps, size=(H, W), mode="bilinear", align_corners=False)
            outs.append(ps)

    return torch.mean(torch.stack(outs, dim=0), dim=0)

# -----------------------------
# Evaluation on a DataLoader
# -----------------------------
def evaluate(model, loader, device, use_tta=False, tta_hflip=True, tta_scales=None, nsd_tau=3):
    model.eval()
    inter, union = init_conf(NUM_CLASSES)
    nsd_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    nsd_cnt = np.zeros(NUM_CLASSES, dtype=np.int64)

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval"):
            x = x.to(device)
            y = y.to(device).long()
            if x.dtype != torch.float32:
                x = x.float() / 255.0

            if use_tta:
                logits = tta_forward(model, x, hflip=tta_hflip, scales=tta_scales)
            else:
                logits = model(x)

            # IoU
            update_conf_from_logits(inter, union, logits, y)

            # NSD per class on CPU
            pred_maps = logits.argmax(1).detach().cpu().numpy()
            gt_maps   = y.detach().cpu().numpy()
            for i in range(pred_maps.shape[0]):
                pm = pred_maps[i]
                gm = gt_maps[i]
                for c in range(NUM_CLASSES):
                    p_bin = (pm == c)
                    g_bin = (gm == c)
                    if not (p_bin.any() or g_bin.any()):
                        continue
                    nsd = compute_nsd_binary(p_bin, g_bin, tau=nsd_tau)
                    nsd_sum[c] += nsd
                    nsd_cnt[c] += 1

    per_class_iou, m_iou = finalize_iou(inter, union)
    per_class_nsd = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    for c in range(NUM_CLASSES):
        if nsd_cnt[c] > 0:
            per_class_nsd[c] = nsd_sum[c] / nsd_cnt[c]
    m_nsd = np.nanmean(per_class_nsd) if np.any(~np.isnan(per_class_nsd)) else 0.0
    final = float(m_iou * m_nsd)

    return (per_class_iou, m_iou, per_class_nsd, m_nsd, final)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_root", required=True, help="Folder with video_* for validation")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (*.pt)")
    ap.add_argument("--model", default="unet", choices=["unet","deeplabv3plus"])
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--long_side", type=int, default=640)
    ap.add_argument("--target_w", type=int, default=640)
    ap.add_argument("--target_h", type=int, default=384)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--nsd_tau", type=int, default=3)
    # TTA flags
    ap.add_argument("--tta_hflip", action="store_true")
    ap.add_argument("--tta_scales", default=None, help='Comma list, e.g. "1.0,1.125"')
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset & Loader (same preproc as train.py via SarRarp50Seg)
    va_ds = SarRarp50Seg(
        [args.val_root],
        long_side=args.long_side, target_w=args.target_w, target_h=args.target_h, aug=False,
    )
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    # Model
    model = build_model(args.model, args.encoder).to(device)
    state = torch.load(args.ckpt, map_location=device)
    # support both {"model": ...} and raw state_dict
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd)

    # Parse TTA scales
    scales = None
    if args.tta_scales:
        scales = [float(s.strip()) for s in args.tta_scales.split(",") if s.strip()]

    # ---- Evaluate NO TTA ----
    iou_c0, m_iou0, nsd_c0, m_nsd0, final0 = evaluate(
        model, va_loader, device, use_tta=False, nsd_tau=args.nsd_tau
    )
    print(f"[NO TTA]  mIoU={m_iou0:.3f}  mNSD={m_nsd0:.3f}  Final={final0:.3f}")
    print("  IoU per class:", " | ".join(
        f"{i}:{CLASS_NAMES[i]}={iou_c0[i]:.2f}" if not np.isnan(iou_c0[i]) else f"{i}:{CLASS_NAMES[i]}=NA"
        for i in range(NUM_CLASSES)))
    print("  NSD per class:", " | ".join(
        f"{i}:{CLASS_NAMES[i]}={nsd_c0[i]:.2f}" if not np.isnan(nsd_c0[i]) else f"{i}:{CLASS_NAMES[i]}=NA"
        for i in range(NUM_CLASSES)))

    # ---- Evaluate WITH TTA ----
    iou_c1, m_iou1, nsd_c1, m_nsd1, final1 = evaluate(
        model, va_loader, device, use_tta=True,
        tta_hflip=args.tta_hflip, tta_scales=scales, nsd_tau=args.nsd_tau
    )
    print(f"[TTA]     mIoU={m_iou1:.3f}  mNSD={m_nsd1:.3f}  Final={final1:.3f}")
    print("  IoU per class:", " | ".join(
        f"{i}:{CLASS_NAMES[i]}={iou_c1[i]:.2f}" if not np.isnan(iou_c1[i]) else f"{i}:{CLASS_NAMES[i]}=NA"
        for i in range(NUM_CLASSES)))
    print("  NSD per class:", " | ".join(
        f"{i}:{CLASS_NAMES[i]}={nsd_c1[i]:.2f}" if not np.isnan(nsd_c1[i]) else f"{i}:{CLASS_NAMES[i]}=NA"
        for i in range(NUM_CLASSES)))

if __name__ == "__main__":
    main()
