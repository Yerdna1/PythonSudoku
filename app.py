from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import sys
import traceback

# Import the main function from your existing script
from main import main

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/', methods=['GET'])
def hello():
    return jsonify({'message': 'Sudoku Solver Backend is running!'})

@app.route('/solve-sudoku', methods=['POST'])
def solve_sudoku():
    print("[DEBUG] Received solve-sudoku request")
    
    if 'image' not in request.files:
        print("[ERROR] No image in request")
        return jsonify({'error': 'No image uploaded'}), 400
   
    file = request.files['image']
    
    # Create a temporary directory to save the uploaded image
    os.makedirs('temp', exist_ok=True)
    temp_path = 'temp/uploaded_sudoku.jpg'
    
    # Save the uploaded image
    file.save(temp_path)
    
    try:
        # Process Sudoku using your existing main function
        print(f"[DEBUG] Processing image: {temp_path}")
        
        # Use a list to capture grids (since we can't use nonlocal in this context)
        grids = [None, None]
        
        def capture_grids(orig, solved):
            grids[0] = orig
            grids[1] = solved
        
        # Process the image
        main(temp_path, 
             confidence_threshold=0.8, 
             debug=False, 
             grid_callback=capture_grids)
        
        return jsonify({
            'message': 'Sudoku solved successfully',
            'original_grid': grids[0].tolist() if grids[0] is not None else None,
            'solution': grids[1].tolist() if grids[1] is not None else None,
            'success': True
        })
    except Exception as e:
        print("[ERROR] Exception in solve_sudoku:")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc(),
            'success': False
        }), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    print("[DEBUG] Starting Flask server")
    app.run(host='0.0.0.0', port=5000, debug=True)