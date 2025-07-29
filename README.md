# Visually Impaired Headphone Guidance System

This project uses computer vision to assist visually impaired individuals in navigating their environment. It utilizes object detection with YOLO, perspective transformation, and sliding window techniques to determine navigable paths and provides auditory cues (left, right, straight, or no path) via headphones. Additionally, it sends an emergency message through Telegram.

---

## 🔧 Features

* 🎯 Real-time object detection using YOLO (ONNX model)
* 🔁 Perspective transformation to simulate a top-down view
* 🧠 Path detection using histogram and sliding windows
* 🔈 Audio feedback to indicate direction (left, straight, right, or no path)
* ⚠️ Emergency alert system via Telegram
* 📍 Detection zone annotation using `supervision` for visualization

---

## 📁 Project Structure

```
project/
│
├── asset/
│   ├── model/
│   │   └── best.onnx              # Pretrained YOLO model
│   ├── video/
│   │   └── video_1.mp4            # Input video for testing
│   └── audio/
│       ├── left.mp3
│       ├── right.mp3
│       ├── straight.mp3
│       └── nopath.mp3
│
├── main.py                        # Main script
└── README.md                      # This file
```

---

## 🚀 Installation & Setup

1. **Clone this repository**:

   ```bash
   git clone https://github.com/yourusername/visually-impaired-headphones.git
   cd visually-impaired-headphones
   ```

2. **Install dependencies**:

   ```bash
   pip install opencv-python numpy matplotlib ultralytics playsound supervision
   ```

3. **Download the ONNX model** and place it in `asset/model/best.onnx`.

4. **Set up Telegram Bot**:

   * Create a bot via [@BotFather](https://t.me/botfather) on Telegram.
   * Get your `TELE_TOKEN` and `CHAT_ID` and insert them into the script:

     ```python
     TELE_TOKEN = "your_bot_token"
     CHAT_ID = "your_chat_id"
     ```

---

## ▶️ Usage

Run the script:

```bash
python main.py
```

Press `ESC` to exit.

The system will:

* Capture frames from a video source.
* Detect obstacles using a YOLO model.
* Perform perspective transformation.
* Apply color thresholding and detect navigable paths.
* Provide directional guidance through sound.
* Alert via Telegram when help is needed.

---

## 📷 Visualization

* `Original`: Live video with detection overlay.
* `Transformed`: Bird-eye view for path processing.
* `Path Detection`: Thresholded mask of the path.
* `Sliding Windows`: Visualizes the histogram and detected path.

---

## 🎵 Audio Cues

| Direction     | Audio File     |
| ------------- | -------------- |
| Go Left       | `left.mp3`     |
| Go Right      | `right.mp3`    |
| Go Straight   | `straight.mp3` |
| No Path Found | `nopath.mp3`   |

---

## 📡 Telegram Emergency Alert

When help is needed, a message is automatically sent:

```
Help requested!
```

Make sure your Telegram bot is authorized to message your chat.

---

## 📌 Notes

* The YOLO model should be trained specifically for relevant obstacles.
* Tune HSV threshold values (`l_h`, `l_s`, `l_v`, `u_h`, `u_s`, `u_v`) for better performance in different environments.
* For real-time deployment, consider switching the video source to a live webcam:

  ```python
  vidcap = cv2.VideoCapture(0)
  ```

---

## 🧠 TODOs

* Add hardware integration (e.g., Raspberry Pi with bone conduction headphones)
* Improve real-time processing performance
* Dynamic thresholding for robust performance in varying light conditions

---

## 📜 License

This project is under the MIT License.
