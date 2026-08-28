import cv2
import numpy as np
import os
from pathlib import Path

def remove_water_column(image_gray, threshold=15):
    """
    Dynamically detects and removes the central dark water column by finding 
    the first bottom return (seabed) on the port and starboard sides.
    """
    h, w = image_gray.shape
    center = w // 2
    
    # Calculate average intensity of each column to find the dark central band
    col_means = np.mean(image_gray, axis=0)
    
    # Scan left from center to find the port-side seabed boundary
    left_bound = center
    while left_bound > 0 and col_means[left_bound] < threshold:
        left_bound -= 1
        
    # Scan right from center to find the starboard-side seabed boundary
    right_bound = center
    while right_bound < w - 1 and col_means[right_bound] < threshold:
        right_bound += 1
        
    # Slice the matrix and stitch the port and starboard sides
    port_side = image_gray[:, :left_bound]
    starboard_side = image_gray[:, right_bound:]
    
    return np.hstack((port_side, starboard_side))

def process_sonar_image(input_path, output_path):
    """
    Runs the V0.1 preprocessing pipeline: Water Column Removal -> Denoise -> Contrast
    """
    # 1. Load as grayscale (sonar is single channel)
    raw_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        print(f"Failed to load {input_path}")
        return

    # 2. Remove the empty water column
    stitched_img = remove_water_column(raw_img)
    
    # 3. Denoise (Bilateral filter reduces speckle but preserves sharp shadow edges)
    clean_img = cv2.bilateralFilter(stitched_img, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 4. Enhance contrast using CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    final_img = clahe.apply(clean_img)
    
    cv2.imwrite(output_path, final_img)
    print(f"Success! Saved to {output_path}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    raw_dir = project_root / "data" / "raw"
    
    process_sonar_image(
        str(raw_dir / "Lucinda_van_Valkenburg_07.png"),
        str(output_dir / "processed_lucinda.png"),
    )
    process_sonar_image(
        str(raw_dir / "Monrovia_05.png"),
        str(output_dir / "processed_monrovia.png"),
    )