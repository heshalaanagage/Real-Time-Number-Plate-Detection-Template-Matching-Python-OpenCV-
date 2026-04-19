#NAME : PERERA A.H.M
#INDEX NO : EGT/21/514


import cv2                  #IMPORT OpenCV library
import numpy as np           #IMPORT NumPy for numerical operations

# CONFIG 
VIDEO_FILE = "Traffic Control CCTV.mp4"         #PATH FOR input video file
VIEW_HEIGHT = 720             #RESIZE video height for display
MIN_CONFIDENCE = 0.7          #MIN confidence to accept number plate

TEMPLATE_FILE = "SS_3.png"    #PATH FOR template image
MATCH_THRESHOLD = 0.62        #MINIMUM MATCH SCORE TO ACCEPT DETECTION

#LOAD TEMPLATE
template_gray = cv2.imread(TEMPLATE_FILE, cv2.IMREAD_GRAYSCALE)

if template_gray is None:
    print("Template image not found:", TEMPLATE_FILE)
    exit()

template_gray = cv2.GaussianBlur(template_gray, (3, 3), 0)
#Apply a small Gaussian blur to the template to reduce noise and improve template matching accuracy

#IMAGE ENHANCEMENT 
def enhance_gray(gray):
    #improve contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    #smooth image but keep edgescal
    smooth = cv2.bilateralFilter(enhanced, 5, 75, 75)
    return smooth

# CANDIDATE EXTRACTION 
def extract_plate_candidates(gray):
    enhanced = enhance_gray(gray)

    #Sobel edge detection in X and Y direction
    gx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)

    #calculate gradient magnitude
    gradient_mag = np.hypot(gx, gy)
    gradient_mag = (gradient_mag / (gradient_mag.max() + 1e-6) * 255).astype(np.uint8)

    #binary threshold to highlight strong edges
    _, binary = cv2.threshold(gradient_mag, 40, 255, cv2.THRESH_BINARY)

    #close gaps using morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    #find contours from binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        ratio = w / max(h, 1)

        #remove small and wrong shaped objects
        if w < 40 or h < 12:
            continue
        if not (3 < ratio < 7):
            continue
        if area < 500 or area / (w * h + 1e-6) < 0.6:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if not (4 <= len(approx) <= 6):
            continue

        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "confidence": 0.5
        })

    return candidates

#CANDIDATE VALIDATION
def refine_candidate(gray, plate):
    x, y, w, h = plate["x"], plate["y"], plate["w"], plate["h"]
    roi = gray[y:y+h, x:x+w]

    if roi.size == 0:
        return False

    #edge detection inside plate area
    edges = cv2.Canny(roi, 50, 150)
    edge_density = cv2.countNonZero(edges) / (roi.size + 1)

    #variance check to confirm text presence
    variance = np.var(roi)

    if not (0.02 < edge_density < 0.5):
        return False
    if variance < 100:
        return False

    #improve confidence if plate quality is good
    if edge_density > 0.05:
        plate["confidence"] += 0.1
    if variance > 300:
        plate["confidence"] += 0.1

    plate["confidence"] = min(plate["confidence"], 1.0)
    return True

#TEMPLATE MATCHING
def match_specific_plate(plate_roi_gray):
    #Apply a small Gaussian blur to the grayscale frame to reduce noise 
    #and make template matching more stable
    blurred = cv2.GaussianBlur(plate_roi_gray, (3, 3), 0)

    highest_score = 0

    #MULTI-SCALE TEMPLATE MATCHING
    #Loop through different scales of the template to detect objects of varying sizes
    for scale in np.linspace(0.4, 2.0, 20):  #scale from 40% to 200% of original template
        #Resize the template according to the current scale
        resized_template = cv2.resize(template_gray, None, fx=scale, fy=scale)

        # Skip this scale if the template is larger than the plate ROI
        if (resized_template.shape[0] >= blurred.shape[0] or
            resized_template.shape[1] >= blurred.shape[1]):
            continue

        #Perform template matching using normalized correlation coefficient
        #cv2.matchTemplate(image, templ, method)
        result = cv2.matchTemplate(blurred, resized_template, cv2.TM_CCOEFF_NORMED)
        #returns score -1 to 1, 1 mean perfect match

        _, score, _, _ = cv2.minMaxLoc(result)
        highest_score = max(highest_score, score)

    return highest_score > MATCH_THRESHOLD

#MAIN LOOP 
def run_detector():
    video = cv2.VideoCapture(VIDEO_FILE)

    if not video.isOpened():
        print("Cannot open video:", VIDEO_FILE)
        return

    while True:
        success, frame = video.read()
        if not success:
            break

        scale_factor = VIEW_HEIGHT / frame.shape[0]
        frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), VIEW_HEIGHT))

        # Convert the current video frame from color (BGR) to grayscale,
        # because template matching works on single-channel images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        plates = extract_plate_candidates(gray)
        plates = [p for p in plates if refine_candidate(gray, p)]
        plates = [p for p in plates if p["confidence"] >= MIN_CONFIDENCE]

        for p in plates:
            x, y, w, h = p["x"], p["y"], p["w"], p["h"]
            roi_gray = gray[y:y+h, x:x+w]

            # check specific number plate using template matching
            is_specific = match_specific_plate(roi_gray)

            # red box for specific plate, green box for others
            color = (0, 0, 255) if is_specific else (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Final Detection", frame)

        # EXIT USING KEY
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    # CLEANUP PROGRAM
    video.release()
    cv2.destroyAllWindows()

#START PROGRAM
if __name__ == "__main__":
    run_detector()
