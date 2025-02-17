import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';

class LiveSudokuSolverPage extends StatefulWidget {
  @override
  _LiveSudokuSolverPageState createState() => _LiveSudokuSolverPageState();
}

class _LiveSudokuSolverPageState extends State<LiveSudokuSolverPage> {
  CameraController? _cameraController;
  bool _isProcessing = false;
    File? _sudokuImage;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    if (cameras.isNotEmpty) {
      _cameraController = CameraController(
        cameras[0], 
        ResolutionPreset.medium,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _cameraController!.initialize();
      
      // Start image stream
      _cameraController!.startImageStream((CameraImage image) {
        if (!_isProcessing) {
          _processFrame(image);
        }
      });

      setState(() {});
    }
  }

   Future<void> _loadSudokuImage() async {
    // Load image from assets
    final imagePath = 'assets/images/sudoku.jpg';
    final File imageFile = File(imagePath);
    
    setState(() {
      _sudokuImage = imageFile;
    });

    // Process the Sudoku image
    await _processSudokuImage(imageFile);
  }

   Future<void> _pickSudokuImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    
    if (pickedFile != null) {
      setState(() {
        _sudokuImage = File(pickedFile.path);
      });

      // Process the picked Sudoku image
      await _processSudokuImage(File(pickedFile.path));
    }
  }

  Future<void> _processFrame(CameraImage image) async {
    _isProcessing = true;

    try {
      // Convert camera image to a file
      File imageFile = await _convertCameraImageToFile(image);

      // Send to backend for Sudoku processing
      final response = await _processSudokuImage(imageFile);

      if (response != null) {
        // Handle solved Sudoku
        _displaySudokuSolution(response);
      }
    } catch (e) {
      print('Error processing frame: $e');
    } finally {
      // Allow next frame to be processed
      _isProcessing = false;
    }
  }

  Future<File> _convertCameraImageToFile(CameraImage image) async {
    final WriteBuffer allBytes = WriteBuffer();
    for (Plane plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }
    final bytes = allBytes.done().buffer.asUint8List();

    // Get temporary directory
    final tempDir = await getTemporaryDirectory();
    final file = await File('${tempDir.path}/sudoku_frame.jpg').create();
    await file.writeAsBytes(bytes);

    return file;
  }

  Future<Map<String, dynamic>?> _processSudokuImage(File imageFile) async {
    final url = Uri.parse('http://192.168.0.109:5000/solve-sudoku');
  
    try {
      var request = http.MultipartRequest('POST', url);
      request.files.add(
        await http.MultipartFile.fromPath('image', imageFile.path)
      );

      var response = await request.send();
    
      if (response.statusCode == 200) {
        final responseBody = await response.stream.bytesToString();
        final solvedSudoku = json.decode(responseBody);
      
        if (solvedSudoku['success']) {
          // Handle solved Sudoku
          print('Original Grid: ${solvedSudoku['original_grid']}');
          print('Solution: ${solvedSudoku['solution']}');
          return solvedSudoku;
        } else {
          print('Sudoku solving failed');
        }
      } else {
        print('Failed to solve Sudoku');
      }
    } catch (e) {
      print('Error processing image: $e');
    }
    return null;
  }

  void _displaySudokuSolution(Map<String, dynamic> solution) {
    // Show an overlay or dialog with the Sudoku solution
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Sudoku Solution'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('Original Grid:'),
            Text(solution['original_grid'].toString()),
            SizedBox(height: 10),
            Text('Solved Grid:'),
            Text(solution['solution'].toString()),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: Text('Close'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Sudoku Solver')),
      body: Column(
        children: [
          // Camera Preview
          _cameraController != null && _cameraController!.value.isInitialized
              ? CameraPreview(_cameraController!)
              : CircularProgressIndicator(),
          
          // Buttons
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton(
                onPressed: _loadSudokuImage,
                child: Text('Load Test Sudoku'),
              ),
              SizedBox(width: 10),
              ElevatedButton(
                onPressed: _pickSudokuImage,
                child: Text('Pick Sudoku Image'),
              ),
            ],
          ),

          // Display picked/loaded image if exists
          if (_sudokuImage != null)
            Image.file(
              _sudokuImage!, 
              height: 200, 
              width: 200, 
              fit: BoxFit.contain
            ),
        ],
      ),
    );

    
  }



  @override
  void dispose() {
    _cameraController?.dispose();
    super.dispose();
  }
}