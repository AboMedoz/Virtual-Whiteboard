# Virtual Whiteboard 

A **hand-tracking digital whiteboard** built with [OpenCV](https://opencv.org/) and [MediaPipe](https://developers.google.com/mediapipe).  
It uses your **webcam feed** and tracks your **hand gestures** to draw or erase on the live video feed.  

---

## 📌 Features
- ✋ **Hand tracking** with MediaPipe.
- 🖊️ Draw on the live webcam feed using your **index finger**.
- 👆 **Drawing mode**: Raise **index finger** (others down) → draw black ink.  
- 🧽 **Eraser mode**: Make a **fist (all fingers down)** → erase with a big white brush.  
- ⏹ Quit the program anytime by pressing **`q`**.
- 🔄 Press **`c`** to **clear the canvas**.
- 🔄 Press **`w`** to **switch from camera feed to whiteboard**.

---

## 🛠️ Requirements
Install the required dependencies:

```bash
pip install opencv-python mediapipe numpy
