import cv2
import numpy as np
import math
from collections import deque

# ----------------- CONFIG -----------------

CAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

RECT_WIDTH = 400
RECT_HEIGHT = 300

WARNING_DIST = 250   # increased for 1280x720
DANGER_DIST = 120

# HSV skin range (tunable) - widened a bit
LOWER_HSV = np.array([0, 20, 60])
UPPER_HSV = np.array([25, 255, 255])

# YCrCb skin range (tunable) - slightly relaxed
LOWER_YCrCb = np.array([0, 135, 85])
UPPER_YCrCb = np.array([255, 180, 135])

STATE_HISTORY = 7

# ------------------------------------------


def point_to_rect_dist(px, py, x1, y1, x2, y2):
    dx = max(max(x1 - px, 0), px - x2)
    dy = max(max(y1 - py, 0), py - y2)
    return math.sqrt(dx * dx + dy * dy)


def detect_open_palm(cnt):
    """
    More tolerant open-palm detector.
    Goal: detect most real open hands, not be mathematically perfect.
    """
    area = cv2.contourArea(cnt)
    if area < 3000:   # not too small
        return False

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = h / float(w + 1e-6)

    # Allow more shapes (vertical or slightly horizontal palm)
    if not (0.5 <= aspect_ratio <= 2.0):
        return False

    rect_area = w * h
    extent = area / float(rect_area + 1e-6)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False

    solidity = area / float(hull_area)

    # Open palm: somewhat convex but with gaps between fingers
    if not (0.4 <= extent <= 0.95):
        return False
    if not (0.5 <= solidity <= 0.95):
        return False

    # Convexity defects: at least one valley (fingers)
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 4:
        return False

    defects = cv2.convexityDefects(cnt, hull_indices)
    if defects is None:
        return False

    finger_valleys = 0

    for i in range(defects.shape[0]):
        s, e, f, depth = defects[i, 0]
        if depth < 1200:   # 1200 ~ small valley, >1200 = more significant
            continue

        start = cnt[s][0]
        end   = cnt[e][0]
        far   = cnt[f][0]

        a = np.linalg.norm(end - far)
        b = np.linalg.norm(start - far)
        c = np.linalg.norm(start - end)
        if a * b == 0:
            continue

        angle = np.degrees(np.arccos((a*a + b*b - c*c) / (2*a*b + 1e-6)))

        # valley between fingers tends to be < ~100 deg
        if 10 < angle < 110:
            finger_valleys += 1

    # Be forgiving: just need at least 1 valley
    if finger_valleys < 1:
        return False

    return True


def extract_fingertips(cnt):
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return []

    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        return []

    fingertips = []

    for i in range(defects.shape[0]):
        s, e, f, depth = defects[i, 0]
        start = tuple(cnt[s][0])
        end   = tuple(cnt[e][0])
        # candidates
        fingertips.append(start)
        fingertips.append(end)

    # unique and keep top-most ones
    unique = list(set(fingertips))
    unique.sort(key=lambda p: p[1])   # y small = upper
    return unique[:5]


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    cx = FRAME_WIDTH // 2
    cy = FRAME_HEIGHT // 2
    rx1 = cx - RECT_WIDTH // 2
    ry1 = cy - RECT_HEIGHT // 2
    rx2 = cx + RECT_WIDTH // 2
    ry2 = cy + RECT_HEIGHT // 2

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    state_history = deque(maxlen=STATE_HISTORY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        mask_hsv = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        mask_ycc = cv2.inRange(ycrcb, LOWER_YCrCb, UPPER_YCrCb)

        # IMPORTANT CHANGE: OR instead of AND
        skin_mask = cv2.bitwise_or(mask_hsv, mask_ycc)

        # Remove face + neck + ears region
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 255), 2)
            neck_ears_margin = int(fh * 0.5)
            ear_margin = int(fw * 0.3)
            start_y = max(0, fy - neck_ears_margin)
            end_y   = min(FRAME_HEIGHT, fy + fh + neck_ears_margin)
            start_x = max(0, fx - ear_margin)
            end_x   = min(FRAME_WIDTH, fx + fw + ear_margin)
            skin_mask[start_y:end_y, start_x:end_x] = 0

        # Morph cleanup
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), 0)

        contours, _ = cv2.findContours(
            skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        chosen_hand = None
        hand_center = None
        fingertips = []
        distance = None
        frame_state = "NO HAND"
        rect_color = (0, 255, 0)

        hand_candidates = []
        for cnt in contours:
            if detect_open_palm(cnt):
                hand_candidates.append(cnt)

        if hand_candidates:
            chosen_hand = max(hand_candidates, key=cv2.contourArea)
            cv2.drawContours(frame, [chosen_hand], -1, (255, 0, 0), 2)

            fingertips = extract_fingertips(chosen_hand)
            for tip in fingertips:
                cv2.circle(frame, tip, 6, (0, 0, 255), -1)

            M = cv2.moments(chosen_hand)
            if M["m00"] != 0:
                hx = int(M["m10"] / M["m00"])
                hy = int(M["m01"] / M["m00"])
                hand_center = (hx, hy)
                cv2.circle(frame, hand_center, 10, (0, 255, 255), -1)
                cv2.putText(
                    frame,
                    "HAND",
                    (hx + 10, hy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            if fingertips:
                min_dist = float("inf")
                closest_tip = None
                for tip in fingertips:
                    d = point_to_rect_dist(tip[0], tip[1], rx1, ry1, rx2, ry2)
                    if d < min_dist:
                        min_dist = d
                        closest_tip = tip
                distance = min_dist
                if closest_tip and (rx1 <= closest_tip[0] <= rx2 and ry1 <= closest_tip[1] <= ry2):
                    distance = 0
            elif hand_center:
                distance = point_to_rect_dist(hx, hy, rx1, ry1, rx2, ry2)
                if rx1 <= hx <= rx2 and ry1 <= hy <= ry2:
                    distance = 0

            if distance is not None:
                if distance > WARNING_DIST:
                    frame_state = "SAFE"
                    rect_color = (0, 255, 0)
                elif distance > DANGER_DIST:
                    frame_state = "WARNING"
                    rect_color = (0, 255, 255)
                else:
                    frame_state = "DANGER"
                    rect_color = (0, 0, 255)

        # State smoothing
        state_history.append(frame_state)
        if state_history:
            stable_state = max(set(state_history), key=state_history.count)
        else:
            stable_state = frame_state

        # Draw virtual zone
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), rect_color, 3)

        # HUD
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 40), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"STATE: {stable_state}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        if distance is not None:
            cv2.putText(
                frame,
                f"Dist: {distance:.1f}px",
                (FRAME_WIDTH - 250, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        if stable_state == "DANGER":
            cv2.putText(
                frame,
                "DANGER DANGER",
                (80, FRAME_HEIGHT // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                1.5,
                (0, 0, 255),
                4,
            )

        cv2.imshow("Skin Mask", skin_mask)
        cv2.imshow("Hand Danger System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord("q")]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()