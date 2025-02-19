import 'dart:io';

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:sudoku_live_camera/constant.dart';


class LiveSudokuSolverPage extends ConsumerStatefulWidget {
  const LiveSudokuSolverPage({super.key});

  @override
  ConsumerState<LiveSudokuSolverPage> createState() => _LiveSudokuSolverPageState();

}

class _LiveSudokuSolverPageState extends ConsumerState<LiveSudokuSolverPage> {
  CameraController? _cameraController;
  bool _isProcessing = false;
  File? _sudokuImage;
  static const String URL_PREF_KEY = 'server_url';
  bool _isSolvingActive = false;

  // Add new state variable
  bool _isShowingProcessing = false;
  DateTime _lastProcessedTime = DateTime.now();

  // Add TextEditingController for the server URL
  late TextEditingController _urlController;
  String serverUrl = serverHetzner;

  @override
  void initState() {
    super.initState();
    _loadSavedUrl();
    _urlController = TextEditingController(text: serverUrl);
    _initializeCamera();
  }

  // Load saved URL from SharedPreferences
  Future<void> _loadSavedUrl() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      serverUrl = prefs.getString(URL_PREF_KEY) ?? serverHetzner;
    });
  }

  // Save URL to SharedPreferences
  Future<void> _saveUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(URL_PREF_KEY, url);
  }

  // Add URL validation
  bool _isValidUrl(String url) {
    try {
      final uri = Uri.parse(url);
      return uri.isScheme('http') || uri.isScheme('https');
    } catch (e) {
      return false;
    }
  }

  // Add method to show server configuration dialog
  // Updated server configuration dialog with validation

  void _showNull() {}
  void _showServerConfig() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Server Configuration'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: _urlController,
              decoration: const InputDecoration(
                labelText: 'Server URL',
          //      hintText: serverHetzner,
                border: OutlineInputBorder(),
                helperText: 'Enter the full URL including protocol and port',
              ),
              keyboardType: TextInputType.url,
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              final newUrl = _urlController.text;
              if (_isValidUrl(newUrl)) {
                setState(() {
                  serverUrl = newUrl;
                });
                _saveUrl(newUrl);
                Navigator.pop(context);
              } else {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('Please enter a valid URL'),
                    backgroundColor: Colors.red,
                  ),
                );
              }
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
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
      //_cameraController!.startImageStream((CameraImage image) {
      //   if (!_isProcessing) {
      //     _processFrame(image);
      //   }
      // });

      setState(() {});
    }
  }

  Future<void> _processCurrentFrame() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Camera not ready'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    setState(() {
      _isShowingProcessing = true;
    });

    try {
      final image = await _cameraController!.takePicture();
      final response = await _processSudokuImage(File(image.path));

      if (response != null) {
        _displaySudokuSolution(response);
      }
    } catch (e) {
      print('Error processing frame: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error processing frame: $e'),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      setState(() {
        _isShowingProcessing = false;
      });
    }
  }

  Future<void> _loadSudokuImage() async {
    try {
      // Show dialog to choose test image
      final String? chosen = await showDialog<String>(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: const Text('Choose Test Image'),
            content: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  ListTile(
                    title: const Text('Sudoku: 34'),
                    onTap: () => Navigator.pop(context, 'assets/images/sudoku34.jpg'),
                  ),
                  ListTile(
                    title: const Text('Sudoku: 38'),
                    onTap: () => Navigator.pop(context, 'assets/images/sudoku38.jpg'),
                  ),
                  ListTile(
                    title: const Text('Sudoku: 43'),
                    onTap: () => Navigator.pop(context, 'assets/images/sudoku43.jpg'),
                  ),
                  ListTile(
                    title: const Text('Sudoku: 88'),
                    onTap: () => Navigator.pop(context, 'assets/images/sudoku88.jpg'),
                  ),
                  ListTile(
                    title: const Text('Sudoku: 106'),
                    onTap: () => Navigator.pop(context, 'assets/images/sudoku106.jpg'),
                  ),
                 
                ],
              ),
            ),
          );
        },
      );

      if (chosen == null) return;

      // Get temporary directory
      final tempDir = await getTemporaryDirectory();
      final tempFile = File('${tempDir.path}/test_sudoku.jpg');

      // Copy asset to temporary file
      ByteData data = await rootBundle.load(chosen);
      List<int> bytes = data.buffer.asUint8List();
      await tempFile.writeAsBytes(bytes);

      setState(() {
        _sudokuImage = tempFile;
      });

      // Show loading indicator
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Processing Sudoku image...'),
          duration: Duration(seconds: 1),
        ),
      );

      // Process the Sudoku image
      final response = await _processSudokuImage(tempFile);

      if (response != null) {
        _displaySudokuSolution(response);
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Failed to process Sudoku image'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } catch (e) {
      print('Error loading sudoku image: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error loading image: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
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
    if (!_isSolvingActive) return; // Skip if solving is not active

    // Check if enough time has passed since last processing
    final now = DateTime.now();
    if (now.difference(_lastProcessedTime).inSeconds < 13) {
      return;
    }
    if (_isProcessing || _isShowingProcessing) return;

    _isProcessing = true;
    setState(() {
      _isShowingProcessing = true;
    });

    try {
      // Show processing indicator
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Processing camera frame...'),
          duration: Duration(seconds: 2),
        ),
      );

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
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error processing frame: $e'),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      // Update last processed time
      _lastProcessedTime = now;
      // Allow next frame to be processed
      _isProcessing = false;
      setState(() {
        _isShowingProcessing = false;
      });
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
  final url = Uri.parse(serverUrl);

  try {
    var request = http.MultipartRequest('POST', url);
    request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));

    var response = await request.send();
    
    if (response.statusCode == 200) {
      final responseBody = await response.stream.bytesToString();
      final decoded = json.decode(responseBody);
      
      // Check if success is true (it's inside the main response)
      if (decoded['success'] == true) {  // Changed this line
        List<List<int>> originalGrid = List<List<int>>.from(
          decoded['original_grid'].map((row) => List<int>.from(row))
        );
        List<List<int>> solutionGrid = List<List<int>>.from(
          decoded['solution'].map((row) => List<int>.from(row))
        );

        return {
          'success': true,
          'original_grid': originalGrid,
          'solution': solutionGrid
        };
      } else {
        print('Solving failed: ${decoded['message'] ?? 'No error message'}');
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Sudoku solving failed'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  } catch (e) {
    print('Error processing image: $e');
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('Error: $e'),
        backgroundColor: Colors.red,
      ),
    );
  }
  return null;
}
  void _displaySudokuSolution(Map<String, dynamic> solution) {
      print('Displaying solution: $solution');  // Add this line

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Sudoku Solution'),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('Original Grid:', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              _buildSudokuGrid(solution['original_grid']),
              const SizedBox(height: 20),
              const Text('Solved Grid:', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              _buildSudokuGrid(solution['solution']),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  Widget _buildSudokuGrid(List<List<int>> grid) {
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: Colors.black, width: 2),
      ),
      child: Column(
        children: List.generate(9, (i) {
          return Row(
            mainAxisSize: MainAxisSize.min,
            children: List.generate(9, (j) {
              bool isRightBorder = (j + 1) % 3 == 0 && j < 8;
              bool isBottomBorder = (i + 1) % 3 == 0 && i < 8;

              return Container(
                width: 30,
                height: 30,
                decoration: BoxDecoration(
                  border: Border(
                    right: BorderSide(
                      width: isRightBorder ? 2 : 1,
                      color: Colors.black,
                    ),
                    bottom: BorderSide(
                      width: isBottomBorder ? 2 : 1,
                      color: Colors.black,
                    ),
                  ),
                ),
                child: Center(
                  child: Text(
                    grid[i][j] == 0 ? '' : grid[i][j].toString(),
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: grid[i][j] == 0 ? FontWeight.normal : FontWeight.bold,
                    ),
                  ),
                ),
              );
            }),
          );
        }),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Sudoku Solver'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: _showNull,
            //_showServerConfig,
          ),
        ],
      ),
      body: Stack(
        children: [
          Column(
            children: [
              // Camera Preview
              Expanded(
                flex: 2,
                child: _cameraController != null && _cameraController!.value.isInitialized
                    ? Stack(
                        alignment: Alignment.bottomCenter,
                        children: [
                          CameraPreview(_cameraController!),
                          Padding(
                            padding: const EdgeInsets.only(bottom: 16),
                            child: ElevatedButton(
                              onPressed: _processCurrentFrame,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.green,
                                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                              ),
                              child: const Text('Solve Camera Frame'),
                            ),
                          ),
                        ],
                      )
                    : const Center(child: CircularProgressIndicator()),
              ),

              // Test image buttons
              Container(
                padding: const EdgeInsets.symmetric(vertical: 10),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    ElevatedButton(
                      onPressed: _loadSudokuImage,
                      child: const Text('Load Test Sudoku'),
                    ),
                    const SizedBox(width: 10),
                    ElevatedButton(
                      onPressed: _pickSudokuImage,
                      child: const Text('Pick Sudoku Image'),
                    ),
                  ],
                ),
              ),

              // Display picked/loaded image and solve button
              if (_sudokuImage != null)
                Expanded(
                  flex: 2,
                  child: Stack(
                    alignment: Alignment.bottomCenter,
                    children: [
                      Container(
                        padding: const EdgeInsets.all(8),
                        child: Image.file(
                          _sudokuImage!,
                          height: 600,
                          width: 600,
                          fit: BoxFit.contain,
                        ),
                      ),
                     
                    ],
                  ),
                ),
            ],
          ),

          // Processing overlay
          if (_isShowingProcessing)
            Container(
              color: Colors.black54,
              child: const Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    CircularProgressIndicator(
                      color: Colors.white,
                    ),
                    SizedBox(height: 16),
                    Text(
                      'Processing...',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }

  // Add disposal of TextEditingController
  @override
  void dispose() {
    _urlController.dispose();
    _cameraController?.dispose();
    super.dispose();
  }
}
