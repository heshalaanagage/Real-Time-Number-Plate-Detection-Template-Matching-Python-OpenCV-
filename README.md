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

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install opencv-python numpy
python ATTEMPH1_2_EGT21514.py

Press Q to quit
