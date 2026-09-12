
from ultralytics import YOLO

# ─────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────

DATA_YAML  = "data.yaml"    # path to your dataset's data.yaml (comes with Roboflow download)
MODEL      = "yolov8s.pt"   # pretrained base — change to yolov8s.pt for better accuracy
EPOCHS     = 50             # increase to 100 for better results
IMG_SIZE   = 416
BATCH      = 4    # lower to 8 if you get out-of-memory errors

if __name__ == '__main__':
    model = YOLO(MODEL)
 
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name="rps_detector",
        patience=20,
        device="0",
        workers=2,       # limits CPU worker processes
        cache=False,      # use GPU
    )

print("\n[DONE] Training complete!")
print("Best model saved at: runs/detect/rps_detector/weights/best.pt")
