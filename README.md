## Snooker Ball Tracking

This project leverages computer vision to track snooker balls in a video using a YOLOv8 object detection model. The script processes a given video, detects snooker balls, and outputs a video with bounding boxes drawn around the balls.

### Features
	•	Detects snooker balls in a video.
	•	Uses YOLOv8 for object detection.
	•	Outputs a video with detected balls and confidence scores.

### Dataset

The dataset contains annotated images with labeled snooker balls, which are used to train the YOLOv8 model. The annotations are in YOLO format.
	•	Images: Stored in train/ and val/ directories.
	•	Labels: Stored in train/labels/ and val/labels/ directories.

### Requirements

#### Python Dependencies
	•	Python 3.7+
#### Libraries:
	•	ultralytics
	•	opencv-python
	•	numpy

Install dependencies

To install the required libraries, run:

    pip install -r requirements.txt

#### Training the Model

1. Download the Dataset

Download the dataset using the following link:

    https://universe.roboflow.com/adadd/gezi-i7n9n/dataset/1/images?split=valid

2. Prepare the data.yaml File

3. Train the YOLOv8 Model

Prepare a train.py script and run it.

This will train the model and save the weights in the runs/train/exp/weights/ directory.

### Running the Detection Script

1. Modify the Script

In scripts/snooker_detection.py, update the following paths:

    VIDEO_PATH = "../videos/snooker_video4.mp4"  # Your video path
    OUTPUT_PATH = "../output/snooker_output.mp4"
    MODEL_PATH = "../runs/train/exp/weights/best.pt"  # Path to your trained model

2. Run the Script

To detect snooker balls in a video, run the following command:

    python scripts/snooker_detection.py

3. Output

The processed video with detected balls will be saved to the location specified in OUTPUT_PATH.

License

This project is licensed under the MIT License. See the LICENSE file for details.
