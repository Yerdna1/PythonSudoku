from logging import debug
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import order_points
from visualization import show_image, show_processing_steps

def preprocess_image(image_path):
    print("[DEBUG] Starting image preprocessing...")
    processing_steps = {}
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Could not read image")
        return None
    print(f"[DEBUG] Original image shape: {img.shape}")
    print(f"[DEBUG] Image type: {type(img)}")
    print(f"[DEBUG] Image dtype: {img.dtype}")

    original_img = img.copy()

     # Resize if image is too large
    max_dimension = 5000
    height, width = img.shape[:2]
    if height > max_dimension or width > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        print(f"[DEBUG] Resizing image from {width}x{height} to {new_width}x{new_height}")
        img = cv2.resize(img, (new_width, new_height))
    
    try:
       # show_image("Original Image", img)
        processing_steps['01-Original Image'] = img
       
    except Exception as e:
        print(f"[WARNING] Could not display image: {str(e)}")
        # Continue processing even if display fails


      # Convert to grayscale with explicit error handling
        print("[DEBUG] Attempting color conversion to grayscale...")
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print("[DEBUG] Color conversion successful")
            else:
                gray = img
                print("[DEBUG] Image already in grayscale")
            
            print(f"[DEBUG] Grayscale image shape: {gray.shape}")
            print(f"[DEBUG] Grayscale dtype: {gray.dtype}")
            
        except Exception as e:
            print(f"[ERROR] Color conversion failed: {str(e)}")
            print("[DEBUG] Attempting alternative conversion method...")
            try:
                # Alternative method: manual conversion
                gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
                gray = gray.astype(np.uint8)
                print("[DEBUG] Alternative conversion successful")
            except Exception as e2:
                print(f"[ERROR] Alternative conversion also failed: {str(e2)}")
                return None


    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #show_image("Grayscale Image", gray)
    processing_steps['02-Grayscale Image'] = gray

    print(f"[DEBUG] Grayscale shape: {gray.shape}")
    print(f"[DEBUG] Grayscale dtype: {gray.dtype}")
    print(f"[DEBUG] Gray value range: {gray.min()} to {gray.max()}")
        
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    #show_image("Blurred Image", blur)
    print("[DEBUG] Applied Gaussian blur")
    processing_steps['03-Blurred Image'] = blur

       # Try different thresholding methods
    methods = [
        ("ADAPTIVE_GAUSSIAN", lambda img: cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
        ("ADAPTIVE_MEAN", lambda img: cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)),
        ("OTSU", lambda img: cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    ]


    # Threshold the image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    #show_image("Thresholded Image", thresh)
    print("[DEBUG] Applied adaptive thresholding")
    processing_steps['04-Thresholded Image'] = thresh

    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[DEBUG] Found {len(contours)} contours")
    
    # Draw all contours
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
  
    #show_image("All Contours", contour_img)
    processing_steps['05-All Contours'] = contour_img
    
    max_area = 0
    grid_contour = None
    
    # Find the largest contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            grid_contour = contour
    
    if grid_contour is None:
        print("[ERROR] No grid contour found!")
        return None
    
    print(f"[DEBUG] Largest contour area: {max_area}")
    
    # Draw the grid contour
    grid_img = img.copy()
    cv2.drawContours(grid_img, [grid_contour], -1, (0, 255, 0), 2)

    #show_image("06-Detected Grid", grid_img)
    processing_steps['06-Detected Grid'] = grid_img

    
    # Get perspective transform
    peri = cv2.arcLength(grid_contour, True)
    approx = cv2.approxPolyDP(grid_contour, 0.02 * peri, True)
    
    if len(approx) == 4:
        pts = np.float32([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
        pts = order_points(pts)


            # Calculate grid dimensions
        grid_width = max(
            np.linalg.norm(pts[1] - pts[0]),  # top width
            np.linalg.norm(pts[3] - pts[2])   # bottom width
        )
        grid_height = max(
            np.linalg.norm(pts[2] - pts[1]),  # right height
            np.linalg.norm(pts[3] - pts[0])   # left height
        )
        
        print(f"[DEBUG] Selected grid:")
        print(f"    Width: {grid_width:.0f}px")
        print(f"    Height: {grid_height:.0f}px")
        print(f"    Area ratio: {(grid_width * grid_height) / (width * height):.3f}")
    
        
        # Draw corner points
        corner_img = grid_img.copy()

        for pt in pts:
            cv2.circle(grid_img, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
            processing_steps['07-Grid Corners'] = corner_img


        #show_image("07-Grid Corners", grid_img)
        
        width = max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[3] - pts[2]))
        height = max(np.linalg.norm(pts[2] - pts[1]), np.linalg.norm(pts[3] - pts[0]))
        
        dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        matrix = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(gray, matrix, (int(width), int(height)))
        processing_steps['08-Warped Grid'] = warped

        print("[DEBUG] Successfully warped perspective")
        if debug:
            show_processing_steps(processing_steps)
        print(matrix)
        print(pts)    
        corners = pts
        return original_img, warped, matrix, corners
    
    print("[ERROR] Could not find 4 corners of the grid")
    return None, None, None, None











def show_stages(stages, i, j):
    """Show processing stages for a cell"""
    n_stages = len(stages)
    plt.figure(figsize=(3*n_stages, 3))
    
    for idx, (title, img) in enumerate(stages.items()):
        plt.subplot(1, n_stages, idx + 1)
        plt.title(title)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    
    plt.suptitle(f'Cell ({i},{j}) Processing Stages')
    plt.tight_layout()
    plt.show()


def remove_grid_lines(warped):
    """Remove grid lines from warped image"""
    # Make a copy of the warped image
    cleaned = warped.copy()
    
    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    
    # Apply threshold to isolate black lines
    _, thresh = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Detect vertical lines
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine the lines
    grid_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Remove lines from original image
    cleaned = cv2.bitwise_and(warped, warped, mask=cv2.bitwise_not(grid_lines))
    
    # Normalize the image
    cleaned = cv2.normalize(cleaned, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    return cleaned

def split_into_cells(cleaned_grid):
    """Split the grid into 81 cells"""
    height, width = cleaned_grid.shape
    cell_height = height // 9
    cell_width = width // 9
    cells = []
    
    # Add some margin to avoid cutting digits
    margin = 5
    
    for i in range(9):
        row = []
        for j in range(9):
            # Calculate cell boundaries with margin
            y1 = max(i * cell_height - margin, 0)
            y2 = min((i + 1) * cell_height + margin, height)
            x1 = max(j * cell_width - margin, 0)
            x2 = min((j + 1) * cell_width + margin, width)
            
            # Extract cell
            cell = cleaned_grid[y1:y2, x1:x2]
            
            # Normalize the cell
            cell = cv2.normalize(cell, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            
            row.append(cell)
        cells.append(row)
    
    return cells    

def process_cell(cell, i, j, debug=True):
    """Process a single cell to extract digit"""
    stages = {}
    stages['Original'] = cell.copy()
    
    # Step 1: Normalize and threshold
    cell_norm = cv2.normalize(cell, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _, binary = cv2.threshold(cell_norm, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    stages['Binary'] = binary
    
    # Step 2: Clean noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    stages['Cleaned'] = cleaned
    
    # Step 3: Find the largest component that's not touching borders
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)
    
    # Filter components
    valid_labels = []
    min_size = cell.shape[0] * cell.shape[1] * 0.03
    max_size = cell.shape[0] * cell.shape[1] * 0.6
    
    for label in range(1, num_labels):  # Skip background
        area = stats[label, cv2.CC_STAT_AREA]
        x, y = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP]
        w, h = stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        
        # Check size and border touching
        if min_size < area < max_size:
            if x > 0 and y > 0 and x + w < binary.shape[1] - 1 and y + h < binary.shape[0] - 1:
                valid_labels.append((label, area))
    
    if valid_labels:
        # Get the largest valid component
        digit_label = max(valid_labels, key=lambda x: x[1])[0]
        
        # Extract digit
        digit_mask = np.zeros_like(binary)
        digit_mask[labels == digit_label] = 255
        stages['Digit'] = digit_mask
        
        # Get bounding box
        x = stats[digit_label, cv2.CC_STAT_LEFT]
        y = stats[digit_label, cv2.CC_STAT_TOP]
        w = stats[digit_label, cv2.CC_STAT_WIDTH]
        h = stats[digit_label, cv2.CC_STAT_HEIGHT]
        
        # Extract and resize digit
        digit = digit_mask[y:y+h, x:x+w]
        
        # Resize to 20x20 maintaining aspect ratio
        target_size = 20
        if w > h:
            new_h = int(h * target_size / w)
            digit = cv2.resize(digit, (target_size, new_h))
            # Center vertically
            top_pad = (target_size - new_h) // 2
            bottom_pad = target_size - new_h - top_pad
            digit = cv2.copyMakeBorder(digit, top_pad, bottom_pad, 0, 0,
                                     cv2.BORDER_CONSTANT, value=0)
        else:
            new_w = int(w * target_size / h)
            digit = cv2.resize(digit, (new_w, target_size))
            # Center horizontally
            left_pad = (target_size - new_w) // 2
            right_pad = target_size - new_w - left_pad
            digit = cv2.copyMakeBorder(digit, 0, 0, left_pad, right_pad,
                                     cv2.BORDER_CONSTANT, value=0)
        
        # Add final padding to reach 28x28
        digit = cv2.copyMakeBorder(digit, 4, 4, 4, 4,
                                 cv2.BORDER_CONSTANT, value=0)
        
        if debug:
            stages['Final'] = digit
            show_stages(stages, i, j)
        
        return digit.astype(np.float32) / 255.0
    
    return np.zeros((28, 28), dtype=np.float32)

def extract_digits(warped, debug=True):
    """Extract all digits from the warped grid"""
    # First remove grid lines
    cleaned_grid = remove_grid_lines(warped)
    
    if debug:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.title('Original Warped')
        plt.imshow(warped, cmap='gray')
        plt.subplot(122)
        plt.title('Grid Lines Removed')
        plt.imshow(cleaned_grid, cmap='gray')
        plt.show()
    
    # Split into cells
    cells = split_into_cells(cleaned_grid)
    
    # Process each cell
    grid = []
    if debug:
        plt.figure(figsize=(15, 15))
    
    for i in range(9):
        row = []
        for j in range(9):
            cell = cells[i][j]
            processed_cell = process_cell(cell, i, j, debug=(debug and i == 5 and j == 3))
            row.append(processed_cell)
            
            if debug:
                plt.subplot(9, 9, i*9 + j + 1)
                plt.imshow(processed_cell, cmap='gray')
                plt.axis('off')
        grid.append(row)
    
    if debug:
        plt.suptitle("Extracted and Processed Cells")
        plt.show()
    
    return np.array(grid)
 