import cv2
import numpy as np
from pathlib import Path

def find_candidate_regions(processed_img):
    """Locates extreme highlights and shadows as candidate debris zones."""
    # Threshold for extreme highlights
    _, highlight_mask = cv2.threshold(processed_img, 200, 255, cv2.THRESH_BINARY)
    
    # Threshold for acoustic shadows 
    _, shadow_mask = cv2.threshold(processed_img, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Combine the masks to find all areas of interest
    combined_mask = cv2.bitwise_or(highlight_mask, shadow_mask)
    
    # Find contours of these extreme regions
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def run_detection(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load {input_path}")
        return
        
    contours = find_candidate_regions(img)
    
    # Convert to BGR so we can draw colored bounding boxes on the grayscale image
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Filter out tiny noise contours and draw boxes around the rest
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 150:  # Ignore small speckles (adjust this threshold as needed)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            count += 1
            
    cv2.imwrite(output_path, output_img)
    print(f"Detected {count} candidate regions. Saved to {output_path}")

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