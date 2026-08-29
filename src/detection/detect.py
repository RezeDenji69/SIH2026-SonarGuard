import cv2
import numpy as np
from pathlib import Path


def find_candidate_regions(processed_img):
    """Locates bright sonar targets while excluding global background artifacts."""
    if processed_img is None or processed_img.size == 0:
        return []

    img_h, img_w = processed_img.shape[:2]
    max_region_area = (img_h * img_w) * 0.25

    _, highlight_mask = cv2.threshold(processed_img, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(highlight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 25:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > max_region_area:
            continue

        roi = processed_img[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        highlight_fraction = np.mean(roi > 180)
        if highlight_fraction < 0.05:
            continue

        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        if h < img_h * 0.9:
            candidate_boxes.append((x1, y1, x2 - x1, y2 - y1))

    return candidate_boxes

def run_detection(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load {input_path}")
        return
        
    grouped_rects = find_candidate_regions(img)
    
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    count = 0
    for (x, y, w, h) in grouped_rects:
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        count += 1
            
    cv2.imwrite(output_path, output_img)
    print(f"Detected {count} unified candidate regions. Saved to {output_path}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "outputs"
    
    run_detection(
        str(output_dir / "processed_lucinda.png"),
        str(output_dir / "detected_lucinda.png")
    )
    run_detection(
        str(output_dir / "processed_monrovia.png"),
        str(output_dir / "detected_monrovia.png")
    )