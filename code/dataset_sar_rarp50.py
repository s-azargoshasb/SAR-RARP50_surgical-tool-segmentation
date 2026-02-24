# dataset_sar_rarp50.py
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------------
# Classes
# -------------------------
CLASS_NAMES = [
    "background", "tool_clasper", "tool_wrist", "tool_shaft",
    "needle", "thread", "suction_tool", "needle_holder", "clamps", "catheter"
]
NUM_CLASSES = len(CLASS_NAMES)

# -------------------------
# Pair discovery
# -------------------------
def _pairs_from_root(root: str):
    pairs = []
    if not os.path.isdir(root):
        return pairs
    for vd in sorted(os.listdir(root)):
        vdir = os.path.join(root, vd)
        fdir = os.path.join(vdir, "frames")
        mdir = os.path.join(vdir, "segmentation")
        if not (os.path.isdir(fdir) and os.path.isdir(mdir)):
            continue
        for mname in sorted(os.listdir(mdir)):
            if not mname.lower().endswith(".png"):
                continue
            img = os.path.join(fdir, mname.replace(".png", ".jpg"))
            msk = os.path.join(mdir, mname)
            if os.path.isfile(img) and os.path.isfile(msk):
                pairs.append((img, msk))
    return pairs

def make_pairs(roots):
    out = []
    for r in roots:
        out.extend(_pairs_from_root(r))
    return out

# very fast prefilter (no decoding)
def _file_ok(p: str, min_bytes: int = 64) -> bool:
    try:
        return os.path.isfile(p) and os.path.getsize(p) >= min_bytes
    except Exception:
        return False

# -------------------------
# Transforms (Albumentations 1.x)
# -------------------------
def get_transforms(long_side, target_w, target_h, aug: bool):
    ts = [
        A.LongestMaxSize(max_size=long_side),
        # constant pad to the target canvas
        A.PadIfNeeded(
            min_height=target_h,
            min_width=target_w,
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),     # image fill (BGR)
            mask_value=0         # mask fill
        ),
    ]
    if aug:
        ts += [
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(0.0, 0.03),
                rotate=(-8, 8),
                shear=(-5, 5),
                mode=cv2.BORDER_CONSTANT,  # Albumentations 1.x
                cval=0,
                cval_mask=0,
                p=0.5,
            ),
            A.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3
            ),
        ]
    ts += [ToTensorV2()]
    return A.Compose(ts)

# -------------------------
# Dataset
# -------------------------
class SarRarp50Seg(Dataset):
    """
    Returns (x, y):
      x: float32 tensor [3,H,W] in [0,1]
      y: long   tensor  [H,W]   with class ids 0..9
    Robustness:
      - Quick prefilter (exists & non-zero size) on first run, cached to JSON.
      - Skips unreadable samples at runtime (no epoch crashes).
    """

    # --------- helpers (self-contained) ---------
    @staticmethod
    def _file_ok(p: str, min_bytes: int = 64) -> bool:
        try:
            return os.path.isfile(p) and os.path.getsize(p) >= min_bytes
        except Exception:
            return False

    @staticmethod
    def _discover_pairs(roots):
        if isinstance(roots, (str, os.PathLike)):
            roots = [roots]
        pairs = []
        for root in roots:
            if not os.path.isdir(root):
                continue
            for vd in sorted(os.listdir(root)):
                vdir = os.path.join(root, vd)
                fdir = os.path.join(vdir, "frames")
                mdir = os.path.join(vdir, "segmentation")
                if not (os.path.isdir(fdir) and os.path.isdir(mdir)):
                    continue
                for mname in sorted(os.listdir(mdir)):
                    if not mname.lower().endswith(".png"):
                        continue
                    img = os.path.join(fdir, mname.replace(".png", ".jpg"))
                    msk = os.path.join(mdir, mname)
                    if os.path.isfile(img) and os.path.isfile(msk):
                        pairs.append((img, msk))
        return pairs

    @staticmethod
    def _build_transforms(long_side, target_w, target_h, aug: bool):
        ts = [
            A.LongestMaxSize(max_size=long_side),
            A.PadIfNeeded(
                min_height=target_h, min_width=target_w,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),   # image fill (BGR)
                mask_value=0       # mask fill
            ),
        ]
        if aug:
            ts += [
                A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent=(0.0, 0.03),
                    rotate=(-8, 8),
                    shear=(-5, 5),
                    mode=cv2.BORDER_CONSTANT,
                    cval=0, cval_mask=0, p=0.5
                ),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3),
            ]
        ts += [ToTensorV2()]
        return A.Compose(ts)

    # --------- main API ---------
    def __init__(self, roots, long_side=960, target_w=960, target_h=544,
                 aug=False, verbose=True, cache_file=None, prefilter_quick=True):
        # auto-pick a cache file near the data (no CLI flag needed)
        if cache_file is None:
            first_root = roots[0] if isinstance(roots, (list, tuple)) else roots
            cache_file = os.path.join(first_root, "_good_pairs.json")

        pairs = None
        # try to load cached good pairs
        if cache_file and os.path.isfile(cache_file):
            try:
                import json
                with open(cache_file, "r") as f:
                    pairs = json.load(f)
                if verbose:
                    print(f"[dataset] loaded {len(pairs)} pairs from cache: {cache_file}")
            except Exception as e:
                if verbose:
                    print(f"[dataset] cache read failed ({e}); will rescan.")

        if pairs is None:
            all_pairs = self._discover_pairs(roots)
            if verbose:
                print(f"[dataset] found {len(all_pairs)} candidate pairs")
            if prefilter_quick:
                # very fast check: existence + non-trivial size (no image decoding)
                pairs = [(im, ms) for (im, ms) in all_pairs if self._file_ok(im) and self._file_ok(ms)]
                if verbose:
                    print(f"[dataset] kept {len(pairs)} after quick prefilter")
            else:
                pairs = all_pairs
            # save cache for next runs
            if cache_file:
                try:
                    import json
                    with open(cache_file, "w") as f:
                        json.dump(pairs, f)
                    if verbose:
                        print(f"[dataset] cached {len(pairs)} pairs -> {cache_file}")
                except Exception as e:
                    if verbose:
                        print(f"[dataset] cache save skipped: {e}")

        if len(pairs) == 0:
            raise RuntimeError("No (image,mask) pairs found. Did you run extract_match_frames.py?")

        self.pairs = pairs
        self.t = self._build_transforms(long_side, target_w, target_h, aug)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Robust: try a few neighbors if one sample is unreadable at runtime
        tries, n = 0, len(self.pairs)
        while tries < 3:
            img_path, msk_path = self.pairs[idx]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)          # BGR uint8
            msk = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)      # uint8 (H,W) or (H,W,3)

            if img is None or msk is None:
                if tries == 0:
                    print(f"[warn] unreadable sample, skipping: {img_path} | {msk_path}")
                idx = (idx + 1) % n
                tries += 1
                continue

            # ensure single-channel mask
            if msk.ndim == 3:
                msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)

            # Albumentations expects RGB np arrays
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            msk = msk.astype(np.uint8, copy=False)

            out = self.t(image=img, mask=msk)
            x = out["image"]     # torch tensor (often uint8)
            y = out["mask"].long()

            # ensure float32 in [0,1] to play nicely with AMP
            if x.dtype != torch.float32:
                x = x.float() / 255.0
            return x, y

        # if we somehow failed 3 times, jump to a random valid index
        ridx = np.random.randint(0, len(self.pairs))
        img_path, msk_path = self.pairs[ridx]
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
        if msk.ndim == 3:
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        out = self.t(image=img, mask=msk.astype(np.uint8, copy=False))
        x = out["image"]; y = out["mask"].long()
        if x.dtype != torch.float32:
            x = x.float() / 255.0
        return x, y
