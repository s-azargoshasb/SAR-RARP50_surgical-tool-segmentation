# Surgical Tool Segmentation 
---

## Pretrained Model

The trained model `unet_resnet34_best.pt` (~280 MB) is too large for GitHub.  
You can download it from [Google Drive](https://drive.google.com/file/d/12WOjzJI3WBJ60vvY6ctJrnuWg2afiR9P/view?usp=sharing) and place it in the `models/` directory:

---

## Repository Structure
```
SAR-RARP50/
│
├── code/
│   ├── dataset_sar_rarp50.py
│   ├── eval_tta.py
│   ├── extract_match_frames.py
│   ├── infer.py
│   ├── train.py
│
├── models/
│   └── unet_resnet34_best.pt   # (you can download it from the provided link)
│
├── results/
│   └── video_42/
│       ├── overlays_no_tta/    # sample frame outputs
│       ├── overlays_tta/       # sample frame outputs
│       └── overlay_tta.mp4     # short demo video
│
├── README.md
└── requirements.txt
```

---

## 1. Setup

Clone the repository and set up a virtual environment:

```bash
git clone <repo-url>
cd SAR-RARP50/code

# Create environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

**Tested on**  
- Google Colab: Python 3.10, PyTorch 2.0.1+cu118, CUDA 11.8 (GPU: A100)  
- Local (Windows): Python 3.13.1, PyTorch 2.8.0 (CPU-only)  

---

## 2. Dataset

Download the dataset from the challenge website and place it in a directory of your choice.  
Example structure:

```
Data/
  train/video_xx/{frames, segmentation}
  val/video_xx/{frames, segmentation}
  test/video_xx/frames
  test_videos/video_xx/video_left.avi
```

Update paths accordingly when running training, inference, or evaluation.

---

## 3. Training

**Model**: UNet (ResNet34 encoder) via `segmentation_models_pytorch`  

**Image sizing**  
- Resize longest side to 640 (aspect ratio preserved)  
- Pad to `target_w=640`, `target_h=384`  

**Loss**  
- Cross-Entropy (+ optional class weights if `--ce_weights` is set)  
- Dice loss  
- Combined: `loss = (1 - dice_weight) * CE + dice_weight * Dice`  

**Augmentations**  
- Affine (scale, translate, rotate, shear)  
- ColorJitter  
- Pad to target canvas  

**Optimization**  
- AdamW (LR=3e-5, no scheduler)  
- Mixed precision (PyTorch AMP) enabled on CUDA  

**Example command:**
```bash
python train.py   --train_roots "Data/train"   --val_roots "Data/val"   --model unet --encoder resnet34   --long_side 640 --target_w 640 --target_h 384   --batch 2 --epochs 40 --lr 3e-5   --dice_weight 0.7   --ce_weights "1,1,1,1,1.2,1.5,1,1,3.0,1.2"   --save_all --show_per_class   --out models/unet_resnet34
```

The best checkpoint is stored at:
```
models/unet_resnet34_best.pt
```

---

## 4. Inference

### A) Inference on frames
```bash
python infer.py   --ckpt models/unet_resnet34_best.pt   --video_dir Data/test/video_42   --out_dir results/test/video_42/overlays_tta   --model unet --encoder resnet34   --long_side 640 --target_w 640 --target_h 384   --tta_hflip --tta_scales "0.875,1.0,1.125"   --save_overlays
```

### B) Inference on video (MP4 output)
```bash
python infer.py   --ckpt models/unet_resnet34_best.pt   --video_file Data/test_videos/video_42/video_left.avi   --out_mp4 results/test_videos/video_42/overlay_tta.mp4   --model unet --encoder resnet34   --long_side 640 --target_w 640 --target_h 384   --tta_hflip --tta_scales "0.875,1.0,1.125"   --overlay_alpha 0.35 --with_legend 1   --stride 1 --num 0
```

---

## 5. Evaluation

To evaluate the model on test data:
```bash
TEST_ROOT=Data/test
CKPT=models/unet_resnet34_best.pt

# Recommended evaluation (with TTA)
python eval_tta.py   --val_root "$TEST_ROOT"   --ckpt "$CKPT"   --model unet --encoder resnet34   --long_side 640 --target_w 640 --target_h 384   --batch 2 --num_workers 2 --nsd_tau 3   --tta_hflip --tta_scales "0.875,1.0,1.125"
```

---

## 6. Results

**Validation/Test Metrics**

| Setting               | mIoU | mNSD | Final |
|------------------------|------|------|-------|
| No TTA                 | 0.702 | 0.603 | 0.423 |
| TTA (0.875,1.0,1.125)  | **0.710** | **0.624** | **0.443** |
| TTA (1.0,1.125,1.25)   | 0.703 | 0.620 | 0.436 |

**Per-class observations**  
- Strong performance on background, shaft, catheter  
- Needle, thread, needle_holder benefit from TTA  
- Clamps remain the most challenging (IoU ~0.30)  

---

## 7. Notes

- CE + Dice loss (with class weights) was important for thin/small tool parts  
- TTA with slight downscale `(0.875,1.0,1.125)` gave the best performance  
- Code is modular: `train.py`, `infer.py`, and `eval_tta.py`  

---

## References

- SAR-RARP50 challenge report:  
    Psychogyios, Dimitrios, et al. "Sar-rarp50: Segmentation of surgical instrumentation and action recognition on robot-assisted radical prostatectomy challenge.
    [Link to paper](https://arxiv.org/html/2401.00496v2)
