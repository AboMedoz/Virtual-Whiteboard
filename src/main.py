import cv2
import mediapipe as mp
import numpy as np

from helpers import check_palette_selection, draw_palette

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

canvas = None
show_whiteboard = False

# Color palette (BGR)
colors = [(0, 0, 255),   # Red
          (0, 255, 0),   # Green
          (255, 0, 0),   # Blue
          (0, 0, 0)]     # Black
color_names = ["Red", "Green", "Blue", "Black"]
current_color = (0, 0, 0)  # default black

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)  # white canvas

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    mode_text = ""  # text to show current mode

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            it, ip = hand_landmarks.landmark[8], hand_landmarks.landmark[6]   # index tip/pip
            mt, mp_ = hand_landmarks.landmark[12], hand_landmarks.landmark[10]  # middle tip/pip
            rt, rp = hand_landmarks.landmark[16], hand_landmarks.landmark[14]  # ring tip/pip
            pt, pp = hand_landmarks.landmark[20], hand_landmarks.landmark[18]  # pinky tip/pip

            x, y = int(it.x * w), int(it.y * h)

            selected = check_palette_selection(x, y, colors)
            if selected:
                current_color = selected
                mode_text = f"Color changed"

            # Draw Mode (index up, pinky down)
            if it.y < ip.y and pt.y > pp.y:
                cv2.circle(canvas, (x, y), 12, current_color, -1)
                mode_text = "Drawing"

            # Erase Mode (all fingers down = fist)
            elif it.y > ip.y and mt.y > mp_.y and rt.y > rp.y and pt.y > pp.y:
                cv2.circle(canvas, (x, y), 40, (255, 255, 255), -1)
                mode_text = "Erasing"

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if show_whiteboard:
        display = canvas.copy()  # copy canvas so tracker isn't permanent
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                it = hand_landmarks.landmark[8]
                x, y = int(it.x * w), int(it.y * h)
                cv2.circle(display, (x, y), 10, (128, 0, 128), 2)  # tracker on COPY
    else:
        ink_mask = np.any(canvas != 255, axis=-1)
        frame[ink_mask] = canvas[ink_mask]
        display = frame
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                it = hand_landmarks.landmark[8]
                x, y = int(it.x * w), int(it.y * h)
                cv2.circle(display, (x, y), 10, (128, 0, 128), 2)  # tracker on live feed

    if mode_text:
        cv2.putText(display, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    draw_palette(display, colors)

    cv2.imshow("Virtual Whiteboard", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        canvas[:] = 255
    if key == ord('s'):
        show_whiteboard = not show_whiteboard

cap.release()
cv2.destroyAllWindows()
