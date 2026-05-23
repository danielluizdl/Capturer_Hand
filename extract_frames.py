"""Extrai frames a 10fps do vídeo e salva crops de cada mesa em frames/."""
import cv2
import os
import sys

VIDEO_PATH  = "video_cortado_1min.mp4"
OUTPUT_DIR  = "frames"
TARGET_FPS  = 10.0
JPEG_QUALITY = 82

TABLE_REGIONS = [
    (0,    0,   960,  540),
    (960,  0,  1920,  540),
    (0,   540,  960, 1080),
    (960, 540, 1920, 1080),
]
TABLE_USEFUL_H = [540, 540, 502, 502]
TABLE_NAMES    = ["mesa0_HL3458", "mesa1_HL4017", "mesa2_HL2332", "mesa3_HL3048"]


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERRO: não foi possível abrir {VIDEO_PATH}")
        sys.exit(1)

    native_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip         = max(1, round(native_fps / TARGET_FPS))
    real_fps     = native_fps / skip

    print(f"Vídeo: {total_frames} frames @ {native_fps:.1f}fps → extraindo a {real_fps:.1f}fps (skip={skip})")

    for name in TABLE_NAMES:
        os.makedirs(os.path.join(OUTPUT_DIR, name), exist_ok=True)

    frame_num  = 0
    saved      = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % skip == 0:
            ts_ms = int(frame_num / native_fps * 1000)
            for tid, (name, (x1, y1, x2, y2), uh) in enumerate(
                zip(TABLE_NAMES, TABLE_REGIONS, TABLE_USEFUL_H)
            ):
                crop = frame[y1: y1 + uh, x1:x2]
                fname = f"f{frame_num:05d}_t{ts_ms:06d}ms.jpg"
                path  = os.path.join(OUTPUT_DIR, name, fname)
                cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            saved += 1
            if saved % 100 == 0:
                pct = frame_num / total_frames * 100
                print(f"  {saved} frames salvos  ({pct:.0f}%)")

        frame_num += 1

    cap.release()
    total_imgs = saved * 4
    print(f"\nConcluído: {saved} frames × 4 mesas = {total_imgs} imagens salvas em '{OUTPUT_DIR}/'")
    return saved


if __name__ == "__main__":
    main()
