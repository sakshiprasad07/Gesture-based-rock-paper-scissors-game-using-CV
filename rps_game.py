import cv2
import time
from collections import deque
from ultralytics import YOLO

# ─────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────

MODEL_PATH        = r"C:\Users\saksp\runs\detect\rps_detector11\weights\best.pt"
CONF_THRESH       = 0.6
WEBCAM_ID         = 0
STABLE_FRAMES     = 8      # how many consecutive frames same gesture must appear before registering
RESULT_HOLD_SECS  = 2      # how long to show the result banner

# Box colors per class (BGR)
COLORS = {
    "Rock":     (60,  60,  220),
    "Paper":    (50,  200, 50),
    "Scissors": (220, 150, 30),
}

# ─────────────────────────────────────────
#  WHO WINS
# ─────────────────────────────────────────

def get_winner(p1, p2):
    if p1 == p2:
        return "Draw!", (0, 200, 255)
    wins = {
        ("Rock",     "Scissors"): "Player 1 wins!",
        ("Scissors", "Paper"):    "Player 1 wins!",
        ("Paper",    "Rock"):     "Player 1 wins!",
        ("Scissors", "Rock"):     "Player 2 wins!",
        ("Paper",    "Scissors"): "Player 2 wins!",
        ("Rock",     "Paper"):    "Player 2 wins!",
    }
    result = wins.get((p1, p2), "")
    color  = (0, 255, 100) if "1" in result else (0, 100, 255)
    return result, color

# ─────────────────────────────────────────
#  DRAW HELPERS
# ─────────────────────────────────────────

def draw_box(image, label, x1, y1, x2, y2, color):
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.rectangle(image, (x1, max(y1 - th - 12, 0)), (x1 + tw + 8, y1), color, -1)
    cv2.putText(image, label, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

def draw_banner(image, text, color):
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, h - 65), (w, h), (20, 20, 20), -1)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.putText(image, text, ((w - tw) // 2, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

def draw_score(image, p1, p2, draws):
    cv2.rectangle(image, (0, 0), (380, 44), (20, 20, 20), -1)
    cv2.putText(image, f"P1: {p1}    P2: {p2}    Draws: {draws}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

def draw_stability_bar(image, label, ratio, x, y, color):
    bar_w = 160
    filled = int(bar_w * ratio)
    cv2.rectangle(image, (x, y), (x + bar_w, y + 16), (60, 60, 60), -1)
    cv2.rectangle(image, (x, y), (x + filled, y + 16), color, -1)
    cv2.putText(image, label, (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────

if __name__ == '__main__':

    print("[INFO] Loading model...")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded! Opening webcam...")

    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print("[ERROR] Webcam not found. Try WEBCAM_ID = 1")
        exit()

    # Stability buffers — last N detections for each player
    p1_buffer = deque(maxlen=STABLE_FRAMES)
    p2_buffer = deque(maxlen=STABLE_FRAMES)

    p1_score  = 0
    p2_score  = 0
    draws     = 0

    banner_text  = "Show both hands!"
    banner_color = (180, 180, 180)
    banner_time  = 0

    last_counted = ("", "")     # last pair that was scored

    print("\n[READY] Left hand = Player 1  |  Right hand = Player 2")
    print("        Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        results    = model(frame, conf=CONF_THRESH, verbose=False)
        boxes      = results[0].boxes
        detections = []

        for box in boxes:
            cls_id     = int(box.cls[0])
            conf       = float(box.conf[0])
            cls_name   = model.names.get(cls_id, "Unknown")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            detections.append((cx, cls_name, conf, x1, y1, x2, y2))

        detections.sort(key=lambda d: d[0])   # left to right

        # ── Feed buffers ──
        if len(detections) >= 1:
            p1_buffer.append(detections[0][1])
        else:
            p1_buffer.clear()

        if len(detections) >= 2:
            p2_buffer.append(detections[1][1])
        else:
            p2_buffer.clear()

        # ── Draw boxes ──
        for i, (cx, cls_name, conf, x1, y1, x2, y2) in enumerate(detections[:2]):
            color      = COLORS.get(cls_name, (200, 200, 200))
            player_tag = "P1" if i == 0 else "P2"
            draw_box(frame, f"{player_tag}: {cls_name} {conf:.0%}", x1, y1, x2, y2, color)

        # ── Stability bars (visual feedback so player knows when it locks in) ──
        p1_ratio = len(p1_buffer) / STABLE_FRAMES if p1_buffer else 0
        p2_ratio = len(p2_buffer) / STABLE_FRAMES if p2_buffer else 0
        p1_color = COLORS.get(p1_buffer[-1], (150,150,150)) if p1_buffer else (100,100,100)
        p2_color = COLORS.get(p2_buffer[-1], (150,150,150)) if p2_buffer else (100,100,100)
        draw_stability_bar(frame, "P1 lock-in", p1_ratio, 10,  h - 110, p1_color)
        draw_stability_bar(frame, "P2 lock-in", p2_ratio, 200, h - 110, p2_color)

        # ── Check if both buffers are stable (all same gesture) ──
        def stable(buf):
            return len(buf) == STABLE_FRAMES and len(set(buf)) == 1

        if stable(p1_buffer) and stable(p2_buffer):
            p1_gesture = p1_buffer[-1]
            p2_gesture = p2_buffer[-1]
            current_pair = (p1_gesture, p2_gesture)

            # Only score if this is a new round (gestures changed)
            if current_pair != last_counted:
                last_counted = current_pair
                result, color = get_winner(p1_gesture, p2_gesture)
                banner_text   = result
                banner_color  = color
                banner_time   = time.time()

                if "1" in result:
                    p1_score += 1
                elif "2" in result:
                    p2_score += 1
                else:
                    draws += 1

                print(f"  {p1_gesture} vs {p2_gesture}  →  {result}")

        elif len(detections) == 1:
            if time.time() - banner_time > RESULT_HOLD_SECS:
                banner_text  = "Show both hands!"
                banner_color = (180, 180, 180)
        elif len(detections) == 0:
            p1_buffer.clear()
            p2_buffer.clear()
            last_counted = ("", "")
            if time.time() - banner_time > RESULT_HOLD_SECS:
                banner_text  = "Show your hands!"
                banner_color = (180, 180, 180)

        # ── Draw UI ──
        draw_banner(frame, banner_text, banner_color)
        draw_score(frame, p1_score, p2_score, draws)

        cv2.imshow("Rock Paper Scissors  —  Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[GAME OVER]")
    print(f"  Player 1 : {p1_score} wins")
    print(f"  Player 2 : {p2_score} wins")
    print(f"  Draws    : {draws}")