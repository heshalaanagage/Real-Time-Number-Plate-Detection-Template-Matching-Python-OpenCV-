# Real-Time-Number-Plate-Detection-Template-Matching-Python-OpenCV-
Real-time vehicle number plate detection and template matching using Python and OpenCV (no AI, pure image processing).

# 🚗 Real-Time Number Plate Detection (OpenCV)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

A real-time number plate detection system built using **classical image processing techniques** in Python.  
This project avoids AI/deep learning and focuses on efficient computer vision methods for detecting and identifying vehicle number plates.

---

##  Features

-  Real-time number plate detection from video
-  Template matching for specific plate recognition
-  Lightweight (no deep learning required)
-  Confidence-based filtering system
- 🟥🟩 Color-coded detection output

---

##  Tech Stack

- Python  
- OpenCV (`cv2`)  
- NumPy  

---

##  Processing Pipeline

###  Image Processing
- CLAHE (Contrast Enhancement)
- Bilateral Filtering (Noise Reduction)
- Sobel Edge Detection
- Thresholding
- Morphological Operations

###  Detection Logic
- Contour Detection
- Shape Filtering (Aspect Ratio, Area)
- Edge Density & Variance Analysis
- Confidence Scoring

###  Template Matching
- Multi-scale matching (40%–200%)
- Normalized correlation coefficient
- Threshold-based classification

---
## ▶️ How to Run the Project

### 🔧 Prerequisites
Make sure you have the following installed:
- Python 3.x
- pip (Python package manager)

---

### 📥 1. Clone the Repository
```bash
git clone https://github.com/your-username/real-time-number-plate-detection-opencv.git
cd real-time-number-plate-detection-opencv
```

---

### 📦 2. Install Dependencies
```bash
pip install opencv-python numpy
```

---

### 📁 3. Prepare Required Files
Ensure the following files are in the project directory:

- ATTEMPH1_2_EGT21514.py (Main script)
- Traffic Control CCTV.mp4 (Input video)
- SS_3.png (Template image)

If your file names or paths are different, update them in the Python file:

```python
VIDEO_FILE = "Traffic Control CCTV.mp4"
TEMPLATE_FILE = "SS_3.png"
```

---

### ▶️ 4. Run the Program
```bash
python ATTEMPH1_2_EGT21514.py
```

---

### 🎮 5. Controls
- Press Q to exit the program

---

### 🖥️ Output
- A video window will open showing detection results
- Green boxes → General number plates
- Red boxes → Specific number plate (template matched)

---

### ⚠️ Troubleshooting

- Video not opening:
  Check the file path and ensure the video exists

- Template not found:
  Make sure SS_3.png is in the project folder

- Module not found error:
  Run: pip install opencv-python numpy
