# infer.py — unified inference for SAR-RARP50
# • Matches train-time preprocessing: LongestMaxSize + PadIfNeeded, then /255 (no ImageNet norm)
# • Inputs:
#     - Frames mode: --video_dir path/to/video_xx (expects frames/ or images/… inside)
#     - Video mode : --video_file path/to/video_left.avi (or any readable video file)
# • Outputs:
#     - Frames mode: overlays (JPG) and/or masks (PNG) into --out_dir
#     - Video mode : single MP4 overlay to --out_mp4 (+ optional per-frame masks dir)
# • Extras:
#     - TTA: --tta_hflip and --tta_scales "0.875,1.0,1.125"
#     - --num to limit frames, --stride for video subsampling, --show for live preview (video mode only)

import os, cv2, argparse, numpy as np, torch, albumentations as A
import segmentation_models_pytorch as smp
from dataset_sar_rarp50 import NUM_CLASSES

# -----------------------------
# Colors / legend
# -----------------------------
CLASS_NAMES = [
    "background","tool_clasper","tool_wrist","tool_shaft",
    "needle","thread","suction_tool","needle_holder","clamps","catheter"
]
# BGR colors (OpenCV)
CLASS_COLORS = np.array([
    (0,0,0),     # 0 background
    (0,0,255),   # 1 tool_clasper
    (0,255,0),   # 2 tool_wrist
    (255,0,0),   # 3 tool_shaft
    (0,255,255), # 4 needle
    (255,0,255), # 5 thread
    (255,255,0), # 6 suction_tool
    (0,0,128),   # 7 needle_holder
    (0,128,0),   # 8 clamps
    (128,0,0),   # 9 catheter
], dtype=np.uint8)

def draw_legend(overlay, present_ids, alpha=0.8):
    if len(present_ids) == 0:
        return overlay
    pad, box_h, box_w, gap = 10, 24, 24, 6
    text_offset = 8
    font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.5; thickness = 1
    rows = len(present_ids)
    legend_w = 260
    legend_h = pad*2 + rows*(box_h + gap) - gap
    panel = overlay.copy()
    cv2.rectangle(panel, (pad, pad), (pad+legend_w, pad+legend_h), (0,0,0), -1)
    overlay = cv2.addWeighted(panel, alpha, overlay, 1-alpha, 0)
    y = pad + 2
    for cid in present_ids:
        b,g,r = map(int, CLASS_COLORS[cid])
        cv2.rectangle(overlay, (pad+6, y), (pad+6+box_w, y+box_h), (b,g,r), -1)
        label = f"{cid}: {CLASS_NAMES[cid]}"
        cv2.putText(overlay, label, (pad+6+box_w+8, y + box_h - text_offset),
                    font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
        y += box_h + gap
    return overlay

def make_overlay(frame_bgr, mask_uint8, alpha=0.35, with_legend=True):
    H, W = frame_bgr.shape[:2]
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for cid, (b,g,r) in enumerate(CLASS_COLORS):
        m = (mask_uint8 == cid)
        color_mask[...,0][m] = b
        color_mask[...,1][m] = g
        color_mask[...,2][m] = r
    over = cv2.addWeighted(frame_bgr, 1.0, color_mask, alpha, 0)
    if with_legend:
        present = [int(v) for v in np.unique(mask_uint8) if 0 <= v < len(CLASS_NAMES)]
        over = draw_legend(over, present)
    return over

# -----------------------------
# Model build/load
# -----------------------------
def build_model(name="unet", encoder="resnet34"):
    name = name.lower()
    if name == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
    elif name in ("deeplabv3plus","deeplab"):
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
    raise ValueError(f"Unknown model '{name}'.")

def load_model(ckpt_path, model_name, encoder, device):
    model = build_model(model_name, encoder)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        sd = state["model"]
    else:
        sd = state
    model.load_state_dict(sd)
    model.to(device).eval()
    return model

# -----------------------------
# Preprocessing (match train.py): resize/pad, then /255
# -----------------------------
def make_preproc(long_side, target_w, target_h):
    return A.Compose([
        A.LongestMaxSize(max_size=long_side),
        A.PadIfNeeded(min_height=target_h, min_width=target_w, border_mode=cv2.BORDER_CONSTANT),
    ])

def to_tensor_chw_uint8(img):
    x = torch.from_numpy(img)            # HWC uint8
    x = x.permute(2, 0, 1).unsqueeze(0)  # 1x3xH'xW'
    return x

def scale_like_train(x):
    if x.dtype != torch.float32:
        x = x.float() / 255.0
    else:
        if x.max() > 1.5:
            x = x / 255.0
    return x

@torch.no_grad()
def forward_logits(model, x, device):
    x = x.to(device, non_blocking=True)
    with torch.amp.autocast('cuda', enabled=(device == "cuda")):
        return model(x)  # [N,C,H,W]

# -----------------------------
# Frame IO
# -----------------------------
def read_frame_from_dir(frames_dir, idx):
    """Read BGR frame by index. Supports jpg/png/jpeg and zero-padded or plain indices."""
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"):
        for name in (f"{idx:09d}{ext}", f"{idx}{ext}"):
            p = os.path.join(frames_dir, name)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    return None

# -----------------------------
# Per-frame inference (with optional TTA)
# -----------------------------
def infer_one_frame(model, frame_bgr, base_long, target_w, target_h, device,
                    tta_hflip=False, tta_scales=None):
    H, W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    preproc = make_preproc(base_long, target_w, target_h)
    x_base_np = preproc(image=rgb)["image"]
    x_base = scale_like_train(to_tensor_chw_uint8(x_base_np))

    base_logits = forward_logits(model, x_base, device)
    logit_sum = base_logits.clone(); votes = 1

    if tta_hflip:
        x_flip = torch.flip(x_base, dims=[3])
        logits_flip = forward_logits(model, x_flip, device)
        logits_flip = torch.flip(logits_flip, dims=[3])
        logit_sum += logits_flip; votes += 1

    if tta_scales:
        for s in tta_scales:
            if abs(s - 1.0) < 1e-6:
                continue
            long_side = int(round(base_long * s))
            x_s_np = make_preproc(long_side, target_w, target_h)(image=rgb)["image"]
            x_s = scale_like_train(to_tensor_chw_uint8(x_s_np))
            logits_s = forward_logits(model, x_s, device)
            if logits_s.shape[-2:] != base_logits.shape[-2:]:
                logits_s = torch.nn.functional.interpolate(
                    logits_s, size=base_logits.shape[-2:], mode="bilinear", align_corners=False
                )
            logit_sum += logits_s; votes += 1

            if tta_hflip:
                x_sf = torch.flip(x_s, dims=[3])
                logits_sf = forward_logits(model, x_sf, device)
                logits_sf = torch.flip(logits_sf, dims=[3])
                if logits_sf.shape[-2:] != base_logits.shape[-2:]:
                    logits_sf = torch.nn.functional.interpolate(
                        logits_sf, size=base_logits.shape[-2:], mode="bilinear", align_corners=False
                    )
                logit_sum += logits_sf; votes += 1

    pred = (logit_sum / float(votes)).argmax(1)[0].detach().cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
    return pred

# -----------------------------
# Runners
# -----------------------------
def run_frames_mode(model, video_dir, out_dir, long_side, target_w, target_h, device,
                    tta_hflip, tta_scales, indices_mode, save_overlays, save_masks,
                    overlay_alpha, with_legend, num_limit):
    # find frames dir
    cand = [os.path.join(video_dir, n) for n in ("frames","Images","images","rgb")]
    frames_dir = next((p for p in cand if os.path.isdir(p)), None)
    if frames_dir is None:
        raise RuntimeError(f"No frames folder found under {video_dir}. Tried: frames/ Images/ images/ rgb/")

    os.makedirs(out_dir, exist_ok=True)

    # pick indices
    def idx_of(n):
        root, ext = os.path.splitext(n)
        return int(root) if root.isdigit() else None

    frame_list = [idx_of(f) for f in os.listdir(frames_dir)
                  if os.path.splitext(f)[1].lower() in (".jpg",".jpeg",".png")]
    indices = sorted([i for i in frame_list if i is not None])

    if num_limit and num_limit > 0:
        indices = indices[:num_limit]

    print(f"[cfg] FRAMES mode: {len(indices)} frames, TTA flip={tta_hflip} scales={tta_scales}, overlays={save_overlays}, masks={save_masks}")

    saved = 0
    for idx in indices:
        frame = read_frame_from_dir(frames_dir, idx)
        if frame is None: continue

        pred = infer_one_frame(model, frame, long_side, target_w, target_h, device,
                               tta_hflip=tta_hflip, tta_scales=tta_scales)

        if save_masks:
            cv2.imwrite(os.path.join(out_dir, f"{idx:09d}.png"), pred)

        if save_overlays:
            overlay = make_overlay(frame, pred, alpha=overlay_alpha, with_legend=with_legend)
            cv2.imwrite(os.path.join(out_dir, f"{idx:09d}_overlay.jpg"), overlay)

        saved += 1

    print(f"[pred] saved {saved} file(s) -> {out_dir}")

def run_video_file_mode(model, video_file, out_mp4, long_side, target_w, target_h, device,
                        tta_hflip, tta_scales, overlay_alpha, with_legend, stride, num_limit,
                        save_masks_dir=None, show=False):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_file}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    if save_masks_dir:
        os.makedirs(save_masks_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer for {out_mp4}")

    # build index list
    idxs = list(range(0, total, max(1, stride)))
    if num_limit and num_limit > 0:
        idxs = idxs[:num_limit]

    print(f"[cfg] VIDEO mode: {video_file}, frames={len(idxs)}, TTA flip={tta_hflip} scales={tta_scales}, alpha={overlay_alpha}, legend={with_legend}, stride={stride}, show={show}")

    saved = 0
    with torch.no_grad():
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok: continue

            pred = infer_one_frame(model, frame, long_side, target_w, target_h, device,
                                   tta_hflip=tta_hflip, tta_scales=tta_scales)

            if save_masks_dir:
                cv2.imwrite(os.path.join(save_masks_dir, f"{idx:09d}.png"), pred)

            overlay = make_overlay(frame, pred, alpha=overlay_alpha, with_legend=with_legend)
            writer.write(overlay)
            saved += 1

            if show:
                cv2.imshow("Segmentation Overlay", overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"[video] wrote {saved} frame(s) -> {out_mp4}")
    if save_masks_dir:
        print(f"[masks] saved to {save_masks_dir}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    # choose input: frames folder OR single video file
    ap.add_argument("--video_dir", default="", help="Path to video_xx folder (contains frames/ or images/).")
    ap.add_argument("--video_file", default="", help="Path to a raw video file (e.g., video_left.avi).")
    # outputs
    ap.add_argument("--out_dir", default="", help="Output folder for FRAMES mode (overlays/masks).")
    ap.add_argument("--out_mp4", default="", help="Output MP4 path for VIDEO mode.")
    ap.add_argument("--save_masks_dir", default="", help="(VIDEO mode) Optional folder to also save per-frame masks (PNG).")
    # model/config
    ap.add_argument("--model", default="unet", choices=["unet","deeplabv3plus"])
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--long_side", type=int, default=640)
    ap.add_argument("--target_w", type=int, default=640)
    ap.add_argument("--target_h", type=int, default=384)
    # TTA
    ap.add_argument("--tta_hflip", action="store_true")
    ap.add_argument("--tta_scales", default="", help='Comma list like "0.875,1.0,1.125"')
    # frames-mode extras
    ap.add_argument("--indices_mode", default="auto", choices=["auto","all"], help="(kept for compatibility; frames mode enumerates frames)")
    ap.add_argument("--save_overlays", action="store_true", help="(FRAMES mode) Save color overlays (JPG).")
    ap.add_argument("--save_masks", action="store_true", help="(FRAMES mode) Save raw masks (PNG).")
    ap.add_argument("--overlay_alpha", type=float, default=0.35)
    ap.add_argument("--with_legend", type=int, default=1)
    ap.add_argument("--num", type=int, default=0, help="Limit number of frames (0=all).")
    # video-mode extras
    ap.add_argument("--stride", type=int, default=1, help="(VIDEO mode) Process every Nth frame.")
    ap.add_argument("--show", action="store_true", help="(VIDEO mode) Show live overlay window; press 'q' to quit.")
    args = ap.parse_args()

    # sanity selection
    use_dir = bool(args.video_dir)
    use_file = bool(args.video_file)
    if use_dir == use_file:
        raise SystemExit("Specify exactly one: either --video_dir or --video_file.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, args.model, args.encoder, device)

    tta_scales = None
    if args.tta_scales.strip():
        try:
            tta_scales = [float(s) for s in args.tta_scales.split(",")]
        except Exception:
            raise ValueError('Bad --tta_scales format. Example: --tta_scales "0.875,1.0,1.125"')

    if use_dir:
        if not args.out_dir:
            raise SystemExit("--out_dir is required when using --video_dir.")
        # default behavior in frames mode: overlays if neither flag is given
        save_overlays = args.save_overlays or (not args.save_masks)
        save_masks    = args.save_masks
        run_frames_mode(
            model, args.video_dir, args.out_dir,
            args.long_side, args.target_w, args.target_h, device,
            tta_hflip=args.tta_hflip, tta_scales=tta_scales,
            indices_mode=args.indices_mode,
            save_overlays=save_overlays, save_masks=save_masks,
            overlay_alpha=args.overlay_alpha, with_legend=bool(args.with_legend),
            num_limit=args.num
        )
    else:
        if not args.out_mp4:
            raise SystemExit("--out_mp4 is required when using --video_file.")
        run_video_file_mode(
            model, args.video_file, args.out_mp4,
            args.long_side, args.target_w, args.target_h, device,
            tta_hflip=args.tta_hflip, tta_scales=tta_scales,
            overlay_alpha=args.overlay_alpha, with_legend=bool(args.with_legend),
            stride=args.stride, num_limit=args.num,
            save_masks_dir=(args.save_masks_dir or None),
            show=args.show
        )

if __name__ == "__main__":
    main()
