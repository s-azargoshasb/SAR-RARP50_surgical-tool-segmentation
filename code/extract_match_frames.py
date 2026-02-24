import os, cv2, argparse

def extract_matching_frames(video_dir: str):
    """
    Expects:
      video_xx/
        video_left.avi
        segmentation/000000060.png ...
    Creates:
      video_xx/frames/000000060.jpg ...
    """
    vpath = os.path.join(video_dir, "video_left.avi")
    mdir  = os.path.join(video_dir, "segmentation")
    out   = os.path.join(video_dir, "frames")

    if not os.path.isfile(vpath) or not os.path.isdir(mdir):
        print(f"[skip] {video_dir} (missing avi or segmentation)")
        return

    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"[err] Cannot open {vpath}")
        return

    mask_names = sorted([f for f in os.listdir(mdir) if f.endswith(".png")])
    for mn in mask_names:
        frame_num = int(os.path.splitext(mn)[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, frame = cap.read()
        if not ok:
            print(f"[warn] Could not read frame {frame_num} in {vpath}")
            continue
        cv2.imwrite(os.path.join(out, mn.replace(".png",".jpg")), frame)
    cap.release()
    print(f"[done] {video_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder with video_xx subfolders")
    args = ap.parse_args()

    for name in sorted(os.listdir(args.root)):
        vdir = os.path.join(args.root, name)
        if os.path.isdir(vdir) and name.startswith("video_"):
            extract_matching_frames(vdir)
