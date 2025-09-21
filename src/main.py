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

    drawing = False
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            it = hand_landmarks.landmark[8]
            ip = hand_landmarks.landmark[6]
            pt = hand_landmarks.landmark[20]
            pp = hand_landmarks.landmark[18]

            x, y = int(it.x * w), int(it.y * h)

            if it.y < ip.y and pt.y > pp.y:
                cv2.circle(canvas, (x, y), 12, (0, 0, 0), -1)
                drawing = True

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, ink_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)  # ink_mask==255 where ink exists

    frame[ink_mask == 255] = (0, 0, 0)

    if drawing:
        cv2.putText(frame, "Drawing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2)

    cv2.imshow("Virtual Whiteboard", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):  # clear canvas (reset to white)
        canvas[:] = 255

cap.release()
cv2.destroyAllWindows()
