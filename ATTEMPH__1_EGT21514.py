#NAME : PERERA A.H.M
#INDEX NO : EGT/21/514

import cv2                 # import OpenCV library for image and video processing
import numpy as np          # import numpy for math and array operations

#CONFIG 
VIDEO_FILE = "Traffic Control CCTV.mp4"        # input video file path
VIEW_HEIGHT = 720           # resize video height for display
MIN_CONFIDENCE = 0.7        # minimum confidence value to accept plate detection

#IMAGE ENHANCEMENT
def enhance_gray(gray):
    # CLAHE used to improve contrast in grayscale image
    # good for plates with low light or shadow
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    #bilateral filter used to reduce noise
    #but keep edges sharp (important for plate edges)
    smooth = cv2.bilateralFilter(enhanced, 5, 75, 75)

    return smooth            #return enhanced grayscale image

#CANDIDATE EXTRACTION 
def extract_plate_candidates(gray):
    #first enhance grayscale image
    enhanced = enhance_gray(gray)

    #show enhanced grayscale in small window
    cv2.imshow("1 Enhanced Gray", cv2.resize(enhanced, (320, 200)))

    #apply Sobel filter in X direction (vertical edges)
    gx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)

    #apply Sobel filter in Y direction (horizontal edges)
    gy = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)

    #calculate gradient magnitude using both X and Y
    gradient_mag = np.hypot(gx, gy)

    #normalize gradient values to 0-255 range
    gradient_mag = (gradient_mag / (gradient_mag.max() + 1e-6) * 255).astype(np.uint8)

    #show gradient magnitude image
    cv2.imshow("2 Gradient Mag", cv2.resize(gradient_mag, (320, 200)))

    #threshold to get strong edges only
    _, binary = cv2.threshold(gradient_mag, 40, 255, cv2.THRESH_BINARY)

    #show binary threshold image
    cv2.imshow("3 Binary Edge", cv2.resize(binary, (320, 200)))

    #create rectangular kernel for morphology
    #width bigger because plates are horizontal shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))

    #closing operation to join broken edges
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    #show morphology result
    cv2.imshow("4 Morph Close", cv2.resize(binary, (320, 200)))

    #find contours from binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []          #list to store possible plate regions

    for cnt in contours:
        #get bounding box of contour
        x, y, w, h = cv2.boundingRect(cnt)

        area = cv2.contourArea(cnt)     #area of contour
        ratio = w / max(h, 1)            #width to height ratio

        #reject very small objects
        if w < 40 or h < 12:
            continue

        #plate aspect ratio normally between 3 and 7
        if not (3 < ratio < 7):
            continue

        #area and fill ratio check
        if area < 500 or area / (w * h + 1e-6) < 0.6:
            continue

        #approximate contour shape
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        #plate shape usually rectangle (4 to 6 points allowed)
        if not (4 <= len(approx) <= 6):
            continue

        #save valid candidate with initial confidence
        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "confidence": 0.5
        })

    return candidates, binary    #return detected candidates and debug binary image

#CANDIDATE VALIDATION
def refine_candidate(gray, plate):
    # extract plate values
    x, y, w, h = plate["x"], plate["y"], plate["w"], plate["h"]

    H, W = gray.shape        #image H and W

    #avoid going outside image
    x2, y2 = min(x + w, W), min(y + h, H)

    #crop region of interest (ROI)
    roi = gray[y:y2, x:x2]

    if roi.size == 0:
        return False        #invalid ROI

    #apply Canny edge detection
    edges = cv2.Canny(roi, 50, 150)

    #calculate edge density
    edge_density = cv2.countNonZero(edges) / (roi.size + 1)

    #calculate intensity variance
    variance = np.var(roi)

    #reject if edges too less or too much
    if not (0.02 < edge_density < 0.5):
        return False

    #reject flat regions
    if variance < 100:
        return False

    #increase confidence based on quality
    if edge_density > 0.05:
        plate["confidence"] += 0.1

    if variance > 300:
        plate["confidence"] += 0.1

    #limit confidence max to 1.0
    plate["confidence"] = min(plate["confidence"], 1.0)

    return True

#DUPLICATE REMOVAL
def suppress_overlaps(plates):
    #sort plates by confidence (high first)
    plates = sorted(plates, key=lambda p: p["confidence"], reverse=True)

    selected = []  #final selected plates
    blocked = [False] * len(plates)

    for i, p in enumerate(plates):
        if blocked[i]:
            continue

        selected.append(p)

        px, py, pw, ph = p["x"], p["y"], p["w"], p["h"]

        for j in range(i + 1, len(plates)):
            if blocked[j]:
                continue

            q = plates[j]

            #calculate intersection area
            ix1 = max(px, q["x"])
            iy1 = max(py, q["y"])
            ix2 = min(px + pw, q["x"] + q["w"])
            iy2 = min(py + ph, q["y"] + q["h"])

            if ix2 > ix1 and iy2 > iy1:
                overlap = (ix2 - ix1) * (iy2 - iy1)
                union = pw * ph + q["w"] * q["h"] - overlap

                #suppress if overlap too high
                if overlap / (union + 1e-6) > 0.3:
                    blocked[j] = True

    return selected

#DRAWING
def render_plates(frame, plates):
    # draw rectangle for each detected plate
    for p in plates:
        cv2.rectangle(
            frame,
            (p["x"], p["y"]),
            (p["x"] + p["w"], p["y"] + p["h"]),
            (0, 255, 0),     #green color box
            2                #thickness
        )
    return frame

#MAIN LOOP
def run_detector():
    #open video file
    cap = cv2.VideoCapture(VIDEO_FILE)

    if not cap.isOpened():
        print("Error opening video file")
        return

    #create display windows
    cv2.namedWindow("Plate Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Processing View", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #resize frame keeping aspect ratio
        scale = VIEW_HEIGHT / frame.shape[0]
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), VIEW_HEIGHT))

        #convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #show original grayscale frame
        cv2.imshow("0 Grayscale", cv2.resize(gray, (320, 200)))

        #extract raw plate candidates
        raw_plates, debug_view = extract_plate_candidates(gray)

        #validate each candidate
        verified = [p for p in raw_plates if refine_candidate(gray, p)]

        #remove overlapping plates and check confidence
        final_plates = [
            p for p in suppress_overlaps(verified)
            if p["confidence"] >= MIN_CONFIDENCE
        ]

        #show results
        cv2.imshow("Plate Detection", render_plates(frame.copy(), final_plates))
        cv2.imshow("Processing View", cv2.cvtColor(debug_view, cv2.COLOR_GRAY2BGR))

        #press Q or q to exit program
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    #release resources
    cap.release()
    cv2.destroyAllWindows()

#PROGRAM START
if __name__ == "__main__":
    run_detector()
