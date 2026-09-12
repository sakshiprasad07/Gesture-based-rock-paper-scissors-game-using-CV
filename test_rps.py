import cv2
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────

MODEL_PATH  = r"C:\Users\saksp\runs\detect\rps_detector11\weights\best.pt"
TEST_FOLDER = "test/images"
CONF_THRESH = 0.4

# Box colors per class (BGR)
COLORS = {
    "Rock":     (60,  60,  220),
    "Paper":    (50,  200, 50),
    "Scissors": (220, 150, 30),
}

# ─────────────────────────────────────────
#  LOAD TRAINED MODEL
# ─────────────────────────────────────────

if __name__ == '__main__':

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # ─────────────────────────────────────
    #  FIND TEST IMAGES
    # ─────────────────────────────────────

    test_images = list(Path(TEST_FOLDER).glob("*.jpg")) + \
                  list(Path(TEST_FOLDER).glob("*.jpeg")) + \
                  list(Path(TEST_FOLDER).glob("*.png"))

    if not test_images:
        print(f"[ERROR] No images found in '{TEST_FOLDER}'")
        print("  → Make sure TEST_FOLDER points to your test/images directory")
        exit(1)

    print(f"[INFO] Found {len(test_images)} test images\n")

    # ─────────────────────────────────────
    #  RUN ON ALL TEST IMAGES
    # ─────────────────────────────────────

    for img_path in test_images:
        results = model(str(img_path), conf=CONF_THRESH)
        result  = results[0]
        boxes   = result.boxes
        image   = result.orig_img.copy()

        print(f"Image: {img_path.name}")

        if len(boxes) == 0:
            print("  → No detections\n")
        else:
            for box in boxes:
                cls_id     = int(box.cls[0])
                confidence = float(box.conf[0])
                cls_name   = model.names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = COLORS.get(cls_name, (200, 200, 200))

                # Draw box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"{cls_name} {confidence:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
                cv2.putText(image, label, (x1 + 4, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                print(f"  → {cls_name:10s}  {confidence:.1%}")

        print()

        # Resize if image is too large for screen
        h, w = image.shape[:2]
        if max(h, w) > 900:
            scale = 900 / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        # Show image — press any key to go to next
        cv2.imshow(f"RPS Detection — press any key for next", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("[DONE] Finished all test images.")

    # ─────────────────────────────────────
    #  FINAL ACCURACY METRICS
    # ─────────────────────────────────────

    print("\n[INFO] Running final validation metrics...")
    metrics = model.val(data="data.yaml", split="test")
    print(f"\nmAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")