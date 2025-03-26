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

#####################################################################################################################
JE TO FULL STACK APLICKACIA
1, backend: PYTHON 
2, frontend: FLUTTER

Aplikacia si vezme na vstupe obrazok z kamery alebo z obrazkov, je tu trenovanie Neuronovych sieti, a rozpoznavanie cislic z obrazku, alebo kamery, pouzije sa najlepsi model z danych natrenovanych
Vysledkom je aj projekckia do povodneho obrazku

Tehcnologie:

PYTHON
MATPLOTLIB
TENSORFLOW
OPENCV
FLUTTER
NEURAL NETWORKS
PLOTLY
