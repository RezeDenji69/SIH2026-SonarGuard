import cv2
import numpy as np
from pathlib import Path
import sys

# Temporarily append the src directory to path so we can import our detection logic
sys.path.append(str(Path(__file__).resolve().parents[1]))
from detection.detect import find_candidate_regions

def validate_and_score(image, contours):
    """
    Validates candidates by enforcing the physics of side-scan sonar:
    A real target must have BOTH an acoustic highlight and an acoustic shadow.
    """
    valid_candidates = []

    for candidate in contours:
        if isinstance(candidate, tuple) and len(candidate) == 4:
            x, y, w, h = candidate
            area = w * h
        else:
            area = cv2.contourArea(candidate)
            if area < 150:
                continue
            x, y, w, h = cv2.boundingRect(candidate)

        if area < 150:
            continue

        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        highlight_pixels = np.sum(roi > 200)
        shadow_pixels = np.sum(roi < 50)

        total_pixels = w * h
        highlight_ratio = highlight_pixels / total_pixels
        shadow_ratio = shadow_pixels / total_pixels

        if highlight_ratio > 0.02 and shadow_ratio > 0.05:
            confidence = min(99, int((highlight_ratio + shadow_ratio) * 300))
            valid_candidates.append((x, y, w, h, confidence))

    return valid_candidates

def run_validation(processed_path, output_path):
    img = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load {processed_path}")
        return
        
    # 1. Get raw candidate zones from our detection module
    contours = find_candidate_regions(img)
    
    # 2. Filter via acoustic geometry and score
    validated = validate_and_score(img, contours)
    
    # 3. Draw the final UI overlay
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h, conf) in validated:
        # Bounding box
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Data label
        label = f"Target: {conf}%"
        cv2.rectangle(output_img, (x, y-25), (x + len(label)*10, y), (0, 255, 0), -1)
        cv2.putText(output_img, label, (x+2, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
    cv2.imwrite(output_path, output_img)
    print(f"Validated {len(validated)} verified targets. Saved to {output_path}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "outputs"
    
    run_validation(
        str(output_dir / "processed_lucinda.png"),
        str(output_dir / "validated_lucinda.png")
    )
    run_validation(
        str(output_dir / "processed_monrovia.png"),
        str(output_dir / "validated_monrovia.png")
    )