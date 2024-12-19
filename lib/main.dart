import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img_lib;
import 'package:flutter/foundation.dart' show kIsWeb;

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return  MaterialApp(
      debugShowCheckedModeBanner: false,
      home: SplashScreen()
    );
  }
}



class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();

    // Add delayed navigation
    Future.delayed(const Duration(seconds: 5), () {
      // Using Navigator.pushReplacement to replace the splash screen with home page
      // This prevents going back to splash screen when pressing back button
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (context) => const MyHomePage(),
        ),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Image.asset(
        "assets/fruveg.gif",
        fit: BoxFit.fill,
        width: double.infinity,
        height: MediaQuery.of(context).size.height * 1,
      ),
    );
  }
}



class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});
  

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? image;
  late ImagePicker imagePicker;
  late Interpreter interpreter;
  List<String>? labels;
  String resultText = "";

  @override
  void initState() {
    super.initState();
    imagePicker = ImagePicker();
    loadModel();

    // Show alert dialog after a brief delay to ensure the screen is ready
    Future.delayed(Duration.zero, () {
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(15),
            ),
            title:  Text(
              'Welcome to FruVeg!',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: Colors.blue.shade700,
              ),
            ),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'About This App:',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
                const SizedBox(height: 8),
                const Text(
                  'I have made it to learn AI model integration with Flutter. Currently, it can scan the following fruits and vegetables:',
                  style: TextStyle(fontSize: 14),
                ),
                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.green.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: const [
                      Text('• Apple', style: TextStyle(fontSize: 14)),
                      Text('• Banana', style: TextStyle(fontSize: 14)),
                      Text('• Mango', style: TextStyle(fontSize: 14)),
                      Text('• Onion', style: TextStyle(fontSize: 14)),
                      Text('• Potato', style: TextStyle(fontSize: 14)),
                      Text('• Watermelon', style: TextStyle(fontSize: 14)),
                    ],
                  ),
                ),
              ],
            ),
            actions: [
              TextButton(
                style: TextButton.styleFrom(
                  foregroundColor: Colors.green,
                ),
                onPressed: () {
                  Navigator.of(context).pop();
                },
                child: const Text('Got it!'),
              ),
            ],
          );
        },
      );
    });
  }

  Future<void> loadModel() async {
    // Load TFLite model
    interpreter = await Interpreter.fromAsset('assets/ml/fruits.tflite');

    // Load labels from text file
    final labelData = await rootBundle.loadString('assets/ml/labels.txt');
    labels = labelData.split('\n');
  }

  Future<void> captureImage() async {
    XFile? selectedImage = await imagePicker.pickImage(source: ImageSource.camera);
    if (selectedImage != null) {
      setState(() {
        image = File(selectedImage.path);
      });
      processImage();
    }
  }

  Future<void> chooseImage() async {
    XFile? selectedImage = await imagePicker.pickImage(source: ImageSource.gallery);
    if (selectedImage != null) {
      setState(() {
        image = File(selectedImage.path);
      });
      processImage();
    }
  }

  Future<void> processImage() async {
    if (image == null) return;

    // Read and preprocess the image
    final imageData = img_lib.decodeImage(await image!.readAsBytes());
    if (imageData == null) return;

    // Resize image to match model input size
    final resizedImage = img_lib.copyResize(imageData, width: 224, height: 224);

    // Convert image to float32 array and normalize
    var inputArray = List.generate(
      224,
          (y) => List.generate(
        224,
            (x) => List.generate(
          3,
              (c) => resizedImage.getPixel(x, y)[c] / 255.0,
        ),
      ),
    );

    // Get output shape from interpreter
    final outputShape = interpreter.getOutputTensor(0).shape;

    // Create output array with correct shape
    var outputs = List.filled(
        outputShape[0],
        List.filled(outputShape[1], 0.0)
    );

    // Run inference
    interpreter.run([inputArray], outputs);

    // Process results
    var outputList = outputs[0];
    var maxScore = 0.0;
    var maxIndex = 0;

    for (var i = 0; i < outputList.length; i++) {
      if (outputList[i] > maxScore) {
        maxScore = outputList[i];
        maxIndex = i;
      }
    }

    setState(() {
      // Set confidence threshold (e.g., 60%)
      const double confidenceThreshold = 0.60;

      // Check if confidence exceeds threshold
      if (maxScore >= confidenceThreshold) {
        if (maxIndex < labels!.length) {
          resultText = 'Detected: ${labels![maxIndex]} (${(maxScore * 100).toStringAsFixed(2)}%)';
        } else {
          resultText = 'Error: Model output index exceeds available labels';
        }
      } else {
        // Show message when confidence is too low
        resultText = 'This does not appear to be a fruit or vegetable\n(Confidence: ${(maxScore * 100).toStringAsFixed(2)}%)';
      }
    });
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("FRUVEG AI", style: TextStyle(color: Color(0xff233ce6), fontWeight: FontWeight.w700, fontSize: 30),),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            if (image != null)
              Image.file(
                image!,
                height: 300,
                width: 300,
                fit: BoxFit.cover,
              )
            else
              const Icon(Icons.add_photo_alternate_outlined, size: 150),
            const SizedBox(height: 20),
            Text(
              resultText,
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  icon: Icon(Icons.add_photo_alternate_outlined),
                  onPressed: chooseImage,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue.shade700,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.only(topLeft: Radius.circular(20),bottomRight: Radius.circular(20),)
                    ),
                  ),
                  label: Text("Choose Image"),

                ),
                ElevatedButton.icon(
                  icon: Icon(Icons.camera),
                  onPressed: captureImage,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue.shade700,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.only(topLeft: Radius.circular(20),bottomRight: Radius.circular(20),)
                    ),
                  ),
                  label: Text("Capture Image"),

                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    interpreter.close();
    super.dispose();
  }
}