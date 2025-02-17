import cv2
import numpy as np
import tensorflow as tf
from digit_recognition import DigitRecognizer
from image_processing import extract_digits, preprocess_image
from sudoku_solver import solve_sudoku
from visualization import plot_digit_confidence, plot_sudoku_recognition, show_final_result, visualize_solution
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

def main(image_path, confidence_threshold=0.8, debug=False,grid_callback=None):

       # Disable pyplot interactive mode
    import matplotlib.pyplot as plt
    plt.ioff()  # Turn off interactive mode

    print(f"[DEBUG] Processing image: {image_path}")
    
    # Step 1: Initialize the digit recognizer
    recognizer = DigitRecognizer(models_dir='models')
    if not recognizer.load_best_model():
        print("[INFO] No pre-trained model found. Training new models...")
        results = recognizer.train_all_models()
        recognizer.plot_training_results()
    else:
        print("[INFO] Loaded pre-trained model")
    
    # Step 2: Preprocess image
    original_img, warped, transform_matrix, corners = preprocess_image(image_path)

    if warped is None:
        print("[ERROR] Could not find Sudoku grid in image")
        return
    
    # Step 3: Extract digits
    grid = extract_digits(warped)
    if grid is None:
        print("[ERROR] Failed to extract digits from grid")
        return
    
    # Step 4: Recognize digits
    print("[DEBUG] Recognizing digits...")
    sudoku_grid = np.zeros((9, 9), dtype=int)
    confidence_grid = np.zeros((9, 9), dtype=float)
    
    # Process each cell
    for i in range(9):
        for j in range(9):
            # Get digit prediction and confidence
            cell = grid[i][j]
            if np.max(cell) > 0:  # Only process cells that aren't empty
                digit, confidence = recognizer.predict(cell)
                confidence_grid[i][j] = confidence
                
                if confidence > confidence_threshold:
                    sudoku_grid[i][j] = digit
                 #  print(f"[DEBUG] Position ({i}, {j}): Detected {digit} with confidence {confidence:.2f}")
    
    # Step 5: Visualize recognition results
    print("[DEBUG] Plotting recognition confidence...")
    plot_digit_confidence(confidence_grid, sudoku_grid)

        # Plot recognition results with threshold control
    plot_sudoku_recognition(sudoku_grid, confidence_grid, 
                          sudoku_grid, confidence_threshold)
    
    # Step 6: Save original grid and solve
    original_grid = sudoku_grid.copy()
    print("\n[DEBUG] Detected Sudoku grid:")
    print_grid(original_grid)
    
    # Step 7: Solve Sudoku
    print("\n[DEBUG] Solving Sudoku...")
    solution = solve_sudoku(sudoku_grid)
    
    if solution is not None:
          # If a callback is provided, use it to return grids
        if grid_callback:
            grid_callback(original_grid, solution)
        print("\n[DEBUG] Solution found:")
        print_grid(solution)
        visualize_solution(original_grid, solution, confidence_grid)

         # Add projection visualization
        projected = show_final_result(original_img, warped, solution, 
                                    transform_matrix, corners,original_grid)
          # Optionally save the projected result
        result_path = image_path.rsplit('.', 1)[0] + '_solution.jpg'
        cv2.imwrite(result_path, projected)
        print(f"\n[INFO] Saved solution image to: {result_path}")


    else:
        print("[ERROR] Could not solve Sudoku")


      # Instead of plt.show(), save figures if needed
    if debug:
            plt.figure()
            plot_digit_confidence(confidence_grid, sudoku_grid)
            plt.savefig('digit_confidence.png')
            plt.close()

            plt.figure()
            plot_sudoku_recognition(sudoku_grid, confidence_grid, sudoku_grid, confidence_threshold)
            plt.savefig('sudoku_recognition.png')
            plt.close()

            plt.figure()
            visualize_solution(original_grid, solution, confidence_grid)
            plt.savefig('solution_visualization.png')
            plt.close()
        
        # If a callback is provided, use it to return grids
    if grid_callback:
            grid_callback(original_grid, solution)

    return {
            'original_grid': original_grid,
            'solution': solution
        }
   

def print_grid(grid):
    """Print the Sudoku grid in a readable format"""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(grid[i][j], end=" ")
        print()

if __name__ == "__main__":
    import os
    import argparse



    parser = argparse.ArgumentParser(description='Sudoku Solver')
    parser.add_argument('image_path', help='Path to the Sudoku image')
    parser.add_argument('--confidence', type=float, default=0.8,
                      help='Confidence threshold (default: 0.8)')
    parser.add_argument('--debug', action='store_true',
                      help='Show debug visualizations')
    
    args = parser.parse_args()
    
    main(args.image_path, 
                  confidence_threshold=args.confidence,
                  debug=args.debug)
    
 