# Basic usage
python main.py path_to_image.jpg

# With custom confidence threshold
python main.py path_to_image.jpg --confidence 0.7

# With debug visualizations
python main.py path_to_image.jpg --debug

#######################################
# Basic usage
python main.py data\sudoku_images\14.jpg

# With custom confidence threshold
python main.py data\sudoku_images\14.jpg --confidence 0.7

# With debug visualizations
python main.py data\sudoku_images\14.jpg --debug



#######################################
Ako to Spustit?
#######################################
cd BACKEND
python -m venv venv
pip install -r requirements.txt
python main.py data\sudoku_images\12.jpg --debug 

#####################################################################################################################
ako pustit backend? ono to bude bezat, napr tu: Running on http://192.168.31.96:5000
#####################################################################################################################
python app.py


 python main.py data\sudoku_images\12.jpg --debug