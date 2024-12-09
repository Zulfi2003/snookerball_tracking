import cv2
from ultralytics import YOLO

# Paths
VIDEO_PATH = "../videos/snooker_video4.mp4"  # Update this with your video name
OUTPUT_PATH = "../output/snooker_output.mp4"

# # Load YOLO model
# model = YOLO("yolov8s.pt")  # Use yolov8s.pt for better accuracy

# Load the model (best weights)
model = YOLO('../models/runs/detect/train3/weights/best.pt')

def main():
    # Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 detection
        results = model.predict(source=frame, conf=0.5, show=False)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class ID
                label = f"{model.names[cls]}: {conf:.2f}"

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display and save frame
        cv2.imshow("Snooker Detection", frame)
        out.write(frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()