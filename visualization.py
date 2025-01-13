import cv2
from matplotlib.patches import Rectangle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from image_processing import order_points



def show_processing_steps(images_dict):
    """Display multiple processing steps in a single figure"""
    n_images = len(images_dict)
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division
    
    plt.figure(figsize=(20, 5*n_rows))
    
    for idx, (title, img) in enumerate(images_dict.items()):
        plt.subplot(n_rows, n_cols, idx + 1)
        plt.title(title)
        
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def show_image(title, image):
    try:
        plt.figure(figsize=(8, 8))
        plt.title(title)
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_image)
        plt.axis('off')
        plt.show()
        plt.close()  # Close the figure to free memory
    except Exception as e:
        print(f"[WARNING] Error displaying {title}: {str(e)}")
        # Continue processing even if display fails


        
def visualize_solution(original_grid, solution, confidence_grid=None):
    """Enhanced visualization of Sudoku solution"""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Original Grid
    ax1 = plt.subplot(131)
    plot_grid(original_grid, title='Original Grid', ax=ax1)
    
    # Plot 2: Solution
    ax2 = plt.subplot(132)
    plot_grid(solution, original_grid, title='Solution', ax=ax2)
    
    # Plot 3: Solution Check
    ax3 = plt.subplot(133)
    check_solution(solution, original_grid, ax=ax3)
    
    plt.tight_layout()
    plt.show()

def plot_grid(grid, original_grid=None, title='', ax=None):
    """Helper function to plot a single Sudoku grid"""
    if ax is None:
        ax = plt.gca()

    # Create background
    plt.imshow(np.zeros((9, 9)), cmap='binary')
    
    # Add grid lines
    for i in range(10):
        linewidth = 2 if i % 3 == 0 else 0.5
        plt.axhline(y=i-0.5, color='black', linewidth=linewidth)
        plt.axvline(x=i-0.5, color='black', linewidth=linewidth)
    
    # Add numbers
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                if original_grid is None:
                    color = 'blue'
                else:
                    color = 'blue' if grid[i][j] == original_grid[i][j] else 'red'
                plt.text(j, i, str(grid[i][j]), 
                        ha='center', va='center', 
                        color=color, fontsize=12, fontweight='bold')
    
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

def check_solution(solution, original_grid, ax=None):
    """Check if solution is valid and visualize errors"""
    if ax is None:
        ax = plt.gca()
    
    # Create background
    ax.imshow(np.zeros((9, 9)), cmap='binary')
    
    # Add grid lines
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(y=i-0.5, color='black', linewidth=lw)
        ax.axvline(x=i-0.5, color='black', linewidth=lw)
    
    # Check rows, columns, and boxes
    errors = find_solution_errors(solution)
    
    # Add numbers and highlight errors
    for i in range(9):
        for j in range(9):
            if solution[i][j] != 0:
                # Determine cell color
                if (i, j) in errors:
                    color = 'red'  # Error
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, 
                                  fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)
                elif original_grid[i][j] != 0:
                    color = 'blue'  # Original number
                else:
                    color = 'green'  # Correctly filled number
                
                ax.text(j, i, str(solution[i][j]), 
                       ha='center', va='center', 
                       color=color, fontsize=12,
                       fontweight='bold')
    
    ax.set_title(f'Solution Check (Errors: {len(errors)})')
    ax.set_xticks([])
    ax.set_yticks([])    

def find_solution_errors(solution):
    """Find cells that violate Sudoku rules"""
    errors = set()
    
    # Check rows
    for i in range(9):
        row = solution[i]
        errors.update(check_duplicates(row, i, 'row'))
    
    # Check columns
    for j in range(9):
        col = solution[:, j]
        errors.update(check_duplicates(col, j, 'col'))
    
    # Check boxes
    for box_i in range(3):
        for box_j in range(3):
            box = solution[box_i*3:(box_i+1)*3, box_j*3:(box_j+1)*3]
            errors.update(check_box_duplicates(box, box_i, box_j))
    
    return errors    

def check_duplicates(arr, idx, type_):
    """Helper function to check for duplicates in row/column"""
    errors = set()
    numbers = {}
    
    for j, num in enumerate(arr):
        if num != 0:
            if num in numbers:
                if type_ == 'row':
                    errors.add((idx, j))
                    errors.add((idx, numbers[num]))
                else:
                    errors.add((j, idx))
                    errors.add((numbers[num], idx))
            numbers[num] = j
    
    return errors

def check_box_duplicates(box, box_i, box_j):
    """Helper function to check for duplicates in 3x3 box"""
    errors = set()
    numbers = {}
    
    for i in range(3):
        for j in range(3):
            num = box[i, j]
            if num != 0:
                if num in numbers:
                    prev_i, prev_j = numbers[num]
                    errors.add((box_i*3 + i, box_j*3 + j))
                    errors.add((box_i*3 + prev_i, box_j*3 + prev_j))
                numbers[num] = (i, j)
    
    return errors

def plot_digit_confidence(confidence_grid, sudoku_grid):
    """Plot confidence levels for digit recognition"""
    plt.figure(figsize=(10, 8))
    
    # Plot confidence heatmap
    im = plt.imshow(confidence_grid, cmap='viridis')
    plt.colorbar(im, label='Confidence')
    
    # Add grid lines
    for i in range(10):
        linewidth = 2 if i % 3 == 0 else 0.5
        plt.axhline(y=i-0.5, color='white', linewidth=linewidth)
        plt.axvline(x=i-0.5, color='white', linewidth=linewidth)
    
    # Add detected digits
    for i in range(9):
        for j in range(9):
            if sudoku_grid[i][j] != 0:
                plt.text(j, i, str(sudoku_grid[i][j]), 
                        ha='center', va='center', 
                        color='red', fontsize=16,
                        bbox=dict(facecolor='none', edgecolor='white', pad=1))
    
    plt.title('Digit Recognition Confidence')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def plot_sudoku_recognition(original_grid, confidence_grid, sudoku_grid, confidence_threshold=0.8):
    """Enhanced visualization of recognition results"""
    fig = plt.figure(figsize=(20, 8))
    
    # Plot 1: Original numbers with confidence heatmap
    ax1 = plt.subplot(121)
    im = ax1.imshow(confidence_grid, cmap='RdYlGn', vmin=0, vmax=1)
    ax1.set_title('Recognition Confidence Heatmap')
    
    # Add grid lines
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax1.axhline(y=i-0.5, color='black', linewidth=lw)
        ax1.axvline(x=i-0.5, color='black', linewidth=lw)
    
    # Add numbers and confidence values
    for i in range(9):
        for j in range(9):
            if sudoku_grid[i][j] != 0:
                confidence = confidence_grid[i][j]
                # Add number
                color = 'white' if confidence < 0.5 else 'black'
                ax1.text(j, i, str(sudoku_grid[i][j]), 
                        ha='center', va='center', 
                        color=color, fontsize=12, fontweight='bold')
                # Add confidence value
                ax1.text(j, i+0.3, f'{confidence:.2f}', 
                        ha='center', va='center', 
                        color=color, fontsize=8)
                
                # Add border for numbers below threshold
                if confidence < confidence_threshold:
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, 
                                  fill=False, color='red', linewidth=2)
                    ax1.add_patch(rect)
    
    plt.colorbar(im, label='Confidence')
    
    # Plot 2: Statistics and distribution
    ax2 = plt.subplot(122)
    
    # Calculate statistics
    confidences = confidence_grid[sudoku_grid != 0]
    below_threshold = sum(confidences < confidence_threshold)
    total_digits = len(confidences)
    
    # Plot confidence distribution
    ax2.hist(confidences, bins=20, range=(0, 1), 
             color='skyblue', edgecolor='black')
    ax2.axvline(x=confidence_threshold, color='red', 
                linestyle='--', label=f'Threshold ({confidence_threshold})')
    
    # Add statistics text
    stats_text = (
        f'Statistics:\n'
        f'Total Digits: {total_digits}\n'
        f'Below Threshold: {below_threshold}\n'
        f'Average Confidence: {np.mean(confidences):.3f}\n'
        f'Median Confidence: {np.median(confidences):.3f}\n'
        f'Min Confidence: {np.min(confidences):.3f}\n'
        f'Max Confidence: {np.max(confidences):.3f}'
    )
    ax2.text(1.1, 0.95, stats_text, 
             transform=ax2.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    ax2.set_title('Confidence Distribution')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Number of Digits')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()    


def project_solution_to_original(original_img, solution, transform_matrix, corners):
    """Project the solution back onto the original image with error handling"""
    try:
        height, width = original_img.shape[:2]
        print(f"[DEBUG] Processing image {width}x{height}")
        result = original_img.copy()
        
        if corners is None or len(corners) != 4:
            print("[ERROR] Invalid grid points")
            return None
            
        # Get the exact coordinates of the grid corners
        corners = np.float32(corners).reshape(-1, 2)
        tl, tr, br, bl = corners
        
        # Verify corner coordinates
        print("[DEBUG] Grid corners:")
        for i, (x, y) in enumerate([tl, tr, br, bl]):
            if not (0 <= x < width and 0 <= y < height):
                print(f"[ERROR] Corner {i} ({x},{y}) outside image bounds")
                return None
            print(f"    Corner {i}: ({x:.1f}, {y:.1f})")
        
        # Calculate dimensions
        grid_width = max(
            np.linalg.norm(tr - tl),
            np.linalg.norm(br - bl)
        )
        grid_height = max(
            np.linalg.norm(bl - tl),
            np.linalg.norm(br - tr)
        )
        
        print(f"[DEBUG] Grid dimensions: {grid_width:.1f}x{grid_height:.1f}")
        
        # Calculate cell dimensions
        cell_width = grid_width / 9.0
        cell_height = grid_height / 9.0
        
        print(f"[DEBUG] Cell dimensions: {cell_width:.1f}x{cell_height:.1f}")
        
        # Scale font size based on cell size
        font_scale = min(cell_width, cell_height) / 40.0  # Adjusted divisor
        thickness = max(2, int(min(cell_width, cell_height) / 15.0))  # Adjusted thickness
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        print(f"[DEBUG] Font settings: scale={font_scale:.1f}, thickness={thickness}")
        
        # Draw numbers
        for i in range(9):
            for j in range(9):
                if solution[i][j] != 0:
                    # Calculate position
                    fx = j / 9.0
                    fy = i / 9.0
                    
                    # Linear interpolation for position
                    top = tl + (tr - tl) * fx
                    bottom = bl + (br - bl) * fx
                    pos = top + (bottom - top) * fy
                    
                    x, y = int(pos[0]), int(pos[1])
                    
                    if not (0 <= x < width and 0 <= y < height):
                        print(f"[WARN] Position ({x},{y}) outside bounds for cell ({i},{j})")
                        continue
                    
                    number = str(solution[i][j])
                    text_size = cv2.getTextSize(number, font, font_scale, thickness)[0]
                    
                    print(f"[DEBUG] Text position for cell ({i},{j}): ({x},{y})")
                    #print(f"[DEBUG] Text size: {text_size}, Font scale: {font_scale}, Thickness: {thickness}")
                    
                    # Center text
                    text_x = int(x - text_size[0]/2)
                    text_y = int(y + text_size[1]/2)
                    
                    # Add background
                    padding = int(min(cell_width, cell_height) * 0.2)  # Adjusted multiplier
                    bg_pts = np.array([
                        [text_x - padding, text_y - text_size[1] - padding],
                        [text_x + text_size[0] + padding, text_y - text_size[1] - padding],
                        [text_x + text_size[0] + padding, text_y + padding],
                        [text_x - padding, text_y + padding]
                    ], dtype=np.int32)
                    
                   # print(f"[DEBUG] Background padding: {padding}")
                    
                    # Draw semi-transparent background
                  #  overlay = result.copy()
                   # cv2.fillPoly(overlay, [bg_pts], (255, 255, 255))
                  #  result = cv2.addWeighted(overlay, 0.9, result, 0.1, 0)
                    
                    # Draw number
                        # Create a mask for the background
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(mask, [bg_pts], 255)
                    
                    # Create a transparent overlay
                    overlay = np.zeros_like(result, dtype=np.uint8)
                    cv2.fillPoly(overlay, [bg_pts], (255, 255, 255, 128))  # Semi-transparent white
                    
                    # Blend the overlay with the result
                    alpha = 0.0  # Adjust transparency level (0 = fully transparent, 1 = fully opaque)
                    result = cv2.addWeighted(result, 1, overlay, alpha, 0)
                    
                    # Draw number
                    cv2.putText(result, number, (text_x, text_y), font, 
                              font_scale, (0, 100, 0, 255), thickness)
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Projection failed: {str(e)}")
        return None
    

def show_final_result(original_img, warped, solution, transform_matrix, corners):
    """Show original, warped, and solution projected back"""
    try:
        plt.figure(figsize=(14, 7))
        
        # Original image with grid corners
        plt.subplot(131)
        plt.title('Original Image')
        orig_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        plt.imshow(orig_rgb)
        
        # Draw detected grid corners
        corners = np.float32(corners).reshape(-1, 2)
        for point in corners:
            plt.plot(point[0], point[1], 'ro')
        plt.axis('off')
        
        # Warped grid with solution
       # plt.subplot(132)
       # plt.title('Warped Grid with Solution')
       # warped_rgb = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)
       # plt.imshow(warped_rgb)
       # plot_grid(solution, title='', ax=plt.gca())
       # plt.axis('off')
        
        # Project solution back
        plt.subplot(133)
        plt.title('Solution Projected on Original')
        try:
            print("[DEBUG] Projecting solution...")
            projected = project_solution_to_original(original_img.copy(), solution, 
                                                  transform_matrix, corners)
            print("[DEBUG] Projection completed")
            if projected is not None:
                plt.imshow(cv2.cvtColor(projected, cv2.COLOR_BGR2RGB))
            else:
                print("[ERROR] Projection returned None")
                plt.imshow(orig_rgb)  # Show original if projection fails
        except Exception as e:
            print(f"[ERROR] Projection failed: {str(e)}")
            plt.imshow(orig_rgb)  # Show original if projection fails
            
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return projected if projected is not None else original_img
        
    except Exception as e:
        print(f"[ERROR] Visualization failed: {str(e)}")
        return original_img