# train.py
import os
import csv
import time
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import segmentation_models_pytorch as smp
import cv2  # <-- for NSD distance transforms and edges

from dataset_sar_rarp50 import SarRarp50Seg, NUM_CLASSES, CLASS_NAMES

# -----------------------------
# Utils
# -----------------------------
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def dice_loss(logits, target, eps=1e-6):
    """Multi-class Dice loss."""
    n, c, h, w = logits.shape
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=c).permute(0, 3, 1, 2).float()  # [N,C,H,W]
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

def build_model(name="unet", encoder="resnet34"):
    name = name.lower()
    if name == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES,
        )
    elif name in ("deeplabv3plus", "deeplab"):
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES,
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Use 'unet' or 'deeplabv3plus'.")

# -----------------------------
# Metrics (IoU)
# -----------------------------
def init_conf(num_classes):
    inter = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    return inter, union

def update_conf(inter, union, logits, target):
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
# NSD (Normalized Surface Dice)
# -----------------------------
def _binary_boundaries(mask_uint8):
    """
    mask_uint8: 0/1 uint8 array (H,W).
    Returns binary boundary maps (0/1 uint8) for mask.
    """
    # Use Canny on 0/255
    edges = cv2.Canny(mask_uint8 * 255, 100, 200)
    return (edges > 0).astype(np.uint8)

def compute_nsd_binary(pred_bin: np.ndarray, gt_bin: np.ndarray, tau: int = 3) -> float:
    """
    pred_bin, gt_bin: boolean/0-1 numpy arrays (H,W)
    tau: tolerance in pixels
    Returns NSD in [0,1].
    """
    # Ensure uint8 0/1
    p = (pred_bin.astype(np.uint8) > 0).astype(np.uint8)
    g = (gt_bin.astype(np.uint8) > 0).astype(np.uint8)

    # Boundaries
    pb = _binary_boundaries(p)
    gb = _binary_boundaries(g)

    # If both have no boundary, define perfect match (1.0)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0

    # Distance transforms: distance to boundary pixels
    # distanceTransform expects non-zero as foreground; we want distance to boundary,
    # so compute on the inverse of boundary mask.
    dist_p = cv2.distanceTransform((1 - pb).astype(np.uint8), cv2.DIST_L2, 0)
    dist_g = cv2.distanceTransform((1 - gb).astype(np.uint8), cv2.DIST_L2, 0)

    # Count boundary pixels within tolerance
    match_p = (dist_g[pb > 0] <= tau).sum() if pb.sum() > 0 else 0
    match_g = (dist_p[gb > 0] <= tau).sum() if gb.sum() > 0 else 0

    total = int(pb.sum() + gb.sum())
    nsd = (match_p + match_g) / total if total > 0 else 1.0
    return float(nsd)

# -----------------------------
# CSV logging helpers
# -----------------------------
def ensure_csv(path, has_val: bool):
    """Create CSV with header if it doesn't exist."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["epoch", "train_loss"]
    if has_val:
        header += ["val_loss", "mIoU", "mNSD", "Final"]  # <-- added mNSD & Final
        # Per-class IoU columns: IoU_0_background, IoU_1_tool_clasper, ...
        for cid, cname in enumerate(CLASS_NAMES):
            header.append(f"IoU_{cid}_{cname}")
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)

def write_csv_row(path, epoch, train_loss, has_val, val_loss=None, miou=None,
                  per_class_iou=None, mnsd=None, final_score=None):
    row = [epoch, float(train_loss)]
    if has_val:
        row.append(float(val_loss) if val_loss is not None else "")
        row.append(float(miou) if miou is not None else "")
        row.append(float(mnsd) if mnsd is not None else "")
        row.append(float(final_score) if final_score is not None else "")
        if per_class_iou is not None:
            for v in per_class_iou:
                row.append(float(v) if (v == v) else "")
        else:
            row.extend([""] * NUM_CLASSES)

    # Try a few times in case another app (Excel) has the file open
    for _ in range(5):
        try:
            with open(path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            return
        except PermissionError:
            time.sleep(2)

    # Fallback: write to a timestamped file so training never dies
    base, ext = os.path.splitext(path)
    alt = f"{base}_{int(time.time())}{ext}"
    with open(alt, "a", newline="") as f:
        csv.writer(f).writerow(row)
    print(f"[warn] CSV locked; wrote this epoch to {alt}")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_roots", nargs="+", required=True,
                    help="Paths with video_xx folders (must contain frames/ and segmentation/)")
    ap.add_argument("--val_roots", nargs="+", default=None,
                    help="Optional: separate roots used only for validation (video-level holdout)")
    ap.add_argument("--model", default="unet", choices=["unet", "deeplabv3plus"])
    ap.add_argument("--encoder", default="resnet34")

    # Aspect-ratio safe sizing (keep AR, then pad) — must match infer.py
    ap.add_argument("--long_side", type=int, default=960, help="Resize so longest side == this")
    ap.add_argument("--target_w", type=int, default=960, help="Pad width to this (divisible by 32 recommended)")
    ap.add_argument("--target_h", type=int, default=544, help="Pad height to this (divisible by 32 recommended)")

    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20, help="Number of epochs to run this invocation")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_split", type=float, default=0.1,
                    help="Used only if --val_roots is not provided")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--dice_weight", type=float, default=0.5, help="Weight for Dice in CE+Dice [0..1]")
    ap.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 recommended on Windows)")
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="Freeze backbone/encoder weights (train decoder/head only)")
    ap.add_argument("--show_per_class", action="store_true",
                    help="Print per-class IoU after each validation")
    ap.add_argument("--log_csv", default=None,
                    help="CSV path for epoch metrics (default: <out>/metrics.csv)")
    ap.add_argument("--save_all", action="store_true",
                    help="Save a checkpoint at the end of every epoch (epoch_NNN.pt)")
    ap.add_argument("--save_every", type=int, default=0,
                    help="If >0, save a checkpoint every N epochs (in addition to best/last)")

    # Loss & metrics
    ap.add_argument("--ce_weights", default=None,
                    help="Comma list of floats with length == NUM_CLASSES (e.g., 10)")
    ap.add_argument("--nsd_tau", type=int, default=3, help="NSD tolerance (pixels)")
    ap.add_argument("--save_best_by", choices=["miou", "final"], default="final",
                    help="Choose criterion for saving best model")

    args = ap.parse_args()

    # --- device must exist before creating tensors on it ---
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = (device == "cuda")

    # Output / logging
    os.makedirs(args.out, exist_ok=True)
    if args.log_csv is None:
        args.log_csv = os.path.join(args.out, "metrics.csv")

    # --- parse CE class weights and build CE loss ---
    weights = None
    if args.ce_weights:
        w = [float(v) for v in args.ce_weights.split(",")]
        assert len(w) == NUM_CLASSES, f"ce_weights must have {NUM_CLASSES} numbers"
        weights = torch.tensor(w, dtype=torch.float, device=device)
    ce = torch.nn.CrossEntropyLoss(weight=weights)
    if weights is not None:
        print("[ce] class weights:", weights.detach().cpu().numpy().tolist())

    # -----------------------------
    # Dataset(s) & DataLoaders
    # -----------------------------
    if args.val_roots:
        tr_ds = SarRarp50Seg(
            args.train_roots,
            long_side=args.long_side, target_w=args.target_w, target_h=args.target_h, aug=True,
        )
        va_ds = SarRarp50Seg(
            args.val_roots,
            long_side=args.long_side, target_w=args.target_w, target_h=args.target_h, aug=False,
        )
        has_val = True
    else:
        ds = SarRarp50Seg(
            args.train_roots,
            long_side=args.long_side, target_w=args.target_w, target_h=args.target_h, aug=True,
        )
        val_len = int(len(ds) * args.val_split)
        if val_len < 1 and len(ds) > 1:
            val_len = 1
        tr_len = max(1, len(ds) - val_len)
        tr_ds, va_ds = random_split(ds, [tr_len, val_len] if val_len > 0 else [len(ds), 0])
        has_val = (val_len > 0)

    tr_loader = DataLoader(
        tr_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    va_loader = None
    if has_val:
        va_loader = DataLoader(
            va_ds, batch_size=max(1, args.batch // 2), shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )

    # Prepare CSV
    ensure_csv(args.log_csv, has_val)

    # -----------------------------
    # Model / Optim / AMP
    # -----------------------------
    model = build_model(args.model, args.encoder).to(device)

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        import torch.nn as nn
        for m in model.encoder.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    # Resume (model + best; try optimizer/scaler)
    # We'll store last best by mIoU for logging, but selection may use Final
    best_miou = -1.0
    best_final = -1.0
    if args.resume:
        try:
            state = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        stored_iou = float(state.get("iou", 0.0))
        best_miou = stored_iou
        # Keep backward compatibility if older checkpoints don't have 'final'
        best_final = float(state.get("final", -1.0))
        print(f"[resume] loaded {args.resume} (stored best mIoU={stored_iou:.3f})")
        if "opt" in state:
            try:
                opt.load_state_dict(state["opt"])
                for g in opt.param_groups:
                    g["lr"] = args.lr
                print(f"[resume] optimizer state restored (lr forced to {args.lr})")
            except Exception as e:
                print(f"[resume] optimizer restore skipped: {e}")
        if "scaler" in state and scaler.is_enabled():
            try:
                scaler.load_state_dict(state["scaler"])
                print("[resume] scaler state restored")
            except Exception as e:
                print(f"[resume] scaler restore skipped: {e}")

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        tr_loss_sum = 0.0
        n_tr = len(tr_ds)

        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device).long()  # ensure correct dtype for CE
            if x.dtype != torch.float32:
                x = x.float() / 255.0
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                logits = model(x)
                loss_ce = ce(logits, y)
                loss_dc = dice_loss(logits, y)
                loss = (1.0 - args.dice_weight) * loss_ce + args.dice_weight * loss_dc

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            tr_loss_sum += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        tr_loss = tr_loss_sum / max(1, n_tr)

        # ---- val ----
        if has_val and va_loader is not None:
            model.eval()
            inter, union = init_conf(NUM_CLASSES)
            va_loss_sum = 0.0
            n_va = len(va_ds)

            # For NSD aggregation
            nsd_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
            nsd_cnt = np.zeros(NUM_CLASSES, dtype=np.int64)

            with torch.no_grad():
                for x, y in tqdm(va_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                    x = x.to(device)
                    y = y.to(device).long()
                    if x.dtype != torch.float32:
                        x = x.float() / 255.0
                    with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                        logits = model(x)
                        loss_ce = ce(logits, y)
                        loss_dc = dice_loss(logits, y)
                        loss = (1.0 - args.dice_weight) * loss_ce + args.dice_weight * loss_dc
                    va_loss_sum += loss.item() * x.size(0)
                    update_conf(inter, union, logits, y)

                    # --- NSD per-class on CPU numpy ---
                    pred_maps = logits.argmax(1).detach().cpu().numpy()
                    gt_maps = y.detach().cpu().numpy()
                    for i in range(pred_maps.shape[0]):
                        pm = pred_maps[i]
                        gm = gt_maps[i]
                        for c in range(NUM_CLASSES):
                            p_bin = (pm == c)
                            g_bin = (gm == c)
                            if not (p_bin.any() or g_bin.any()):
                                continue  # skip empty class for this sample
                            nsd = compute_nsd_binary(p_bin, g_bin, tau=args.nsd_tau)
                            nsd_sum[c] += nsd
                            nsd_cnt[c] += 1

            per_class_iou, m_iou = finalize_iou(inter, union)
            va_loss = va_loss_sum / max(1, n_va)

            # finalize NSD
            per_class_nsd = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
            for c in range(NUM_CLASSES):
                if nsd_cnt[c] > 0:
                    per_class_nsd[c] = nsd_sum[c] / nsd_cnt[c]
            m_nsd = np.nanmean(per_class_nsd) if np.any(~np.isnan(per_class_nsd)) else 0.0

            final_score = float(m_iou * m_nsd)

            # Choose reference metric for "best"
            if args.save_best_by == "final":
                metric_value = final_score
            else:
                metric_value = m_iou

            print(f"Epoch {epoch}: train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
                  f"mIoU={m_iou:.3f}  mNSD={m_nsd:.3f}  Final={final_score:.3f}")

            if args.show_per_class:
                shown_iou = []
                for cid, name in enumerate(CLASS_NAMES):
                    if not np.isnan(per_class_iou[cid]):
                        shown_iou.append(f"{cid}:{name}={per_class_iou[cid]:.2f}")
                print("  per-class IoU:", " | ".join(shown_iou))

                shown_nsd = []
                for cid, name in enumerate(CLASS_NAMES):
                    if not np.isnan(per_class_nsd[cid]):
                        shown_nsd.append(f"{cid}:{name}={per_class_nsd[cid]:.2f}")
                print("  per-class NSD:", " | ".join(shown_nsd))

        else:
            # No validation: use -train_loss so "improvement" means lower train loss
            metric_value = -tr_loss
            m_iou = 0.0
            m_nsd = 0.0
            final_score = 0.0
            va_loss = 0.0
            per_class_iou = None
            print(f"Epoch {epoch}: train_loss={tr_loss:.4f}  (no validation split)")

        # ---- Save checkpoints (BEST first), then LAST, then OPTIONAL per-epoch ----
        improved = metric_value > (best_final if args.save_best_by == "final" else best_miou)
        if improved:
            if args.save_best_by == "final":
                best_final = metric_value
            else:
                best_miou = metric_value

            ckpt_best = os.path.join(args.out, f"{args.model}_{args.encoder}_best.pt")
            torch.save({
                "model": model.state_dict(),
                "iou": m_iou,
                "final": final_score,
                "args": vars(args),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
            }, ckpt_best)
            tag = (f"best Final={metric_value:.3f}" if args.save_best_by == "final"
                   else f"best mIoU={metric_value:.3f}")
            print(f"[saved] {ckpt_best}  ({tag})")

        ckpt_last = os.path.join(args.out, f"{args.model}_{args.encoder}_last.pt")
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "iou": m_iou,
            "final": final_score,
            "args": vars(args),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
        }

        # always save "last"
        torch.save(state, ckpt_last)

        # always save this epoch
        ckpt_epoch = os.path.join(args.out, f"{args.model}_{args.encoder}_e{epoch:03d}.pt")
        torch.save(state, ckpt_epoch)
        print(f"[saved] {ckpt_epoch}")

        # save best if this is best so far
        if final_score > best_final:
            best_final = final_score
            ckpt_best = os.path.join(args.out, f"{args.model}_{args.encoder}_best.pt")
            torch.save(state, ckpt_best)
            print(f"[best updated] {ckpt_best}")


        # ---- Log to CSV (robust) ----
        if has_val and va_loader is not None:
            write_csv_row(args.log_csv, epoch, tr_loss, True, va_loss, m_iou, per_class_iou, m_nsd, final_score)
        else:
            write_csv_row(args.log_csv, epoch, tr_loss, False)

if __name__ == "__main__":
    main()
