# Sign-Language-Recognition-3D

A real-time sign language recognition system using 3D Convolutional Neural Networks (CNN3D) based on the Inception V3 architecture.

## Important Links
- Pre-trained Model: [Google Drive](https://drive.google.com/drive/u/2/folders/14_gpXuZrT9XtnPTk81D1cNr22mTuou6j)
- Project Demo: [YouTube](https://www.youtube.com/watch?v=DM33agCKVE0)

## Overview

This project implements a real-time sign language recognition system using webcam input. The system processes video frames through a 3D CNN model to recognize and classify sign language gestures. The application provides a web interface for easy interaction and real-time predictions.

## Features

- Real-time video capture from webcam
- Face detection using MTCNN
- 3D CNN model based on Inception V3 for gesture recognition
- Web-based interface for interaction
- Support for 29 different sign language gestures
- Frame sequence processing for accurate predictions

## Prerequisites

- Python 3.x
- Flask
- OpenCV (cv2)
- TensorFlow
- MTCNN
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SignLanguageRecognition3D.git
cd SignLanguageRecognition3D
```

2. Install required dependencies:
```bash
pip install flask opencv-python tensorflow mtcnn numpy
```

3. Download the pre-trained model:
- Access the model files from [Google Drive](https://drive.google.com/drive/u/2/folders/14_gpXuZrT9XtnPTk81D1cNr22mTuou6j)
- Place the model files in the `model29` directory

## Usage

1. Start the application:
```bash
python webstreaming.py
```

2. Access the web interface:
- Open your browser and navigate to `http://127.0.0.1:5000`
- The interface will display the webcam feed

3. Recording and Prediction:
- Click "Start" to begin recording video frames
- The system will capture 120 frames (approximately 4 seconds)
- Click "Predict" to process the recorded sequence
- The system will display top 5 predictions with confidence scores

## Project Structure

```
SignLanguageRecognition3D/
├── webstreaming.py         # Main application file
├── utils/
│   ├── load_tf.py         # TensorFlow model loader
│   ├── videoto3D.py       # Video frame processor
│   ├── FindTop.py         # Top predictions finder
│   ├── cutImage.py        # Image processing utilities
│   └── encodingClass.py   # Class label encoder
├── templates/
│   └── index_ver2.html    # Web interface template
├── model29/               # Pre-trained model directory
└── data/                  # Output directory for processed videos
```

## Technical Details

- Input Processing:
  - Captures 120 frames from webcam
  - Selects 15 frames evenly distributed across the sequence
  - Detects face region using MTCNN
  - Crops and resizes frames to 224x224 pixels

- Model Architecture:
  - Based on Inception V3
  - Modified for 3D convolution to process temporal information
  - Outputs predictions for 29 different sign language classes

## Demo

Watch the demonstration video: [YouTube Demo](https://www.youtube.com/watch?v=DM33agCKVE0)

## Acknowledgments

- Based on the Inception V3 architecture
- Uses MTCNN for face detection
- [Add any other acknowledgments]
