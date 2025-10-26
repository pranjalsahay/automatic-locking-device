🔒 High Precision Lock Scanner

A real-time face/object recognition and encryption system built with Python, OpenCV, and face_recognition.
It captures a live video feed, detects a reference target (face or object), and automatically locks and saves encrypted match images.
The system can also estimate distance and display latency per frame for performance monitoring.

🚀 Features

🎯 Dual recognition mode — works with both faces and arbitrary objects

🔐 Encrypted match saving using AES (via cryptography.fernet)

⚡ Latency monitoring (ms/frame in yellow)

🟩 Displays TARGET FOUND when matched

🟥 Displays TARGET NOT FOUND when reference is loaded but no match detected

📏 Distance estimation via calibrated focal length

🧠 Automatically loads the last locked reference on startup

💾 Optional object size calibration for more accurate distance estimation

🧰 Requirements

Make sure you have Python 3.8+ installed, then install dependencies:

pip install opencv-python face_recognition cryptography numpy


You also need to install dlib (required by face_recognition):

Windows:

pip install dlib


macOS / Linux:

brew install cmake
pip install dlib

🖥️ How It Works

Run the script:

python lock_scanner.py


Controls (keyboard shortcuts):

Key	Function
c	Capture current frame as reference (face/object)
l	Load reference image from disk
r	Reset reference and delete lock
f	Calibrate focal length (for distance estimation)
q	Quit the program

Once a reference is set:

The system continuously checks each frame for matches.

When a match is found:
🟩 TARGET FOUND appears, and a cropped, encrypted match image is saved.

If no match is found:
🟥 TARGET NOT FOUND appears.

Match images are saved in the matches/ directory as encrypted .lock files.

🔑 Encryption Details

The first run generates a secret key stored in secret.key.

Encrypted reference image is saved as locked_reference.lock.

Matches are saved as encrypted .lock files in the matches/ folder.

Encryption uses AES-128 (Fernet symmetric encryption).

🧮 Distance Estimation

If you calibrate the focal length (f key) or enter real-world object width,
the system estimates target distance in centimeters using:

distance = (known_width * focal_length) / perceived_width

📊 Performance Notes
Hardware	Latency	FPS
Laptop CPU	80–150 ms/frame	~7–12 FPS
Desktop CPU	40–80 ms/frame	~12–25 FPS
GPU (CUDA)	20–40 ms/frame	~25–50 FPS

Latency and FPS depend on image size, number of features, and hardware.

📁 File Structure
📂 project/
 ┣ 📜 lock_scanner.py
 ┣ 📜 secret.key
 ┣ 📜 locked_reference.lock
 ┣ 📂 matches/
 ┃ ┗ 📜 match_0.lock
 ┗ 📜 README.md

⚠️ Notes

Make sure your webcam is connected before running.

Always close the program with q to properly release the camera.

Avoid sharing your secret.key — it’s required to decrypt saved matches.

🧑‍💻 Author

Developed by: [Your Name]
