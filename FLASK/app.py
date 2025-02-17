from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

# Import your Sudoku solver functions
from main import process_sudoku_image  # Adjust import based on your project structure

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/solve-sudoku', methods=['POST'])
def solve_sudoku():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
   
    file = request.files['image']
   
    # Read image
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
   
    try:
        # Process Sudoku (use your existing functions)
        result = process_sudoku_image(img)
        
        return jsonify({
            'original_grid': result['original'].tolist() if 'original' in result else None,
            'solved_grid': result['answers'].tolist() if 'answers' in result else None,
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)