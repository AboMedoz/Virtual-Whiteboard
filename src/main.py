import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

canvas = None
show_whiteboard = False

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

            # Draw Mode (index up, pinky down)
            if it.y < ip.y and pt.y > pp.y:
                cv2.circle(canvas, (x, y), 12, (0, 0, 0), -1)
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
                cv2.circle(display, (x, y), 10, (0, 0, 255), 2)  # tracker on COPY
    else:
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, ink_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        frame[ink_mask == 255] = (0, 0, 0)
        display = frame
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                it = hand_landmarks.landmark[8]
                x, y = int(it.x * w), int(it.y * h)
                cv2.circle(display, (x, y), 10, (0, 0, 255), 2)  # tracker on live feed

    if mode_text:
        cv2.putText(display, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

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
