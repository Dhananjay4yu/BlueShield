import cv2
import numpy as np
import os
import time

# Paths to YOLO model files - Update these with your actual paths
weights_path = "yolov4-csp-s-mish.weights"   # Path to YOLO weights
config_path = "yolov4.cfg"                   # Path to YOLO configuration
classes_path = "coco.names"                  # Path to COCO class names

# Check if the model files exist
for path, desc in [(weights_path, "Weights"), (config_path, "Configuration"), (classes_path, "Classes")]:
    if not os.path.exists(path):
        print(f"{desc} file not found: {path}")
        exit()

# Load YOLO model
def load_yolo_model(weights, config, classes):
    """Loads YOLO model with weights, configuration, and class names."""
    net = cv2.dnn.readNet(weights, config)

    # Set preferable backend and target; fallback to default and CPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    with open(classes, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()

    # Ensure compatibility with OpenCV versions for extracting output layers
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except AttributeError:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, class_names, output_layers

# Perform object detection on the given frame
def detect_objects(net, output_layers, class_names, frame):
    """Detects objects in the provided frame using the YOLO model."""
    height, width, _ = frame.shape

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (416, 416))

    blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to filter overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        label = f"{class_names[class_ids[i]]} {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Process real-time video from webcam
def process_realtime_video():
    """Performs object detection in real-time using the webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reducing the frame size for smoother performance
        frame = cv2.resize(frame, (640, 480))

        start_time = time.time()  # Measure processing time for each frame

        processed_frame = detect_objects(net, output_layers, class_names, frame)
        cv2.imshow("Object Detection - Real-time Video", processed_frame)

        # Limiting frame rate for smooth performance
        elapsed_time = time.time() - start_time
        delay = max(1, int((1 / 30 - elapsed_time) * 1000))  # Targeting 30 FPS
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load YOLO model
net, class_names, output_layers = load_yolo_model(weights_path, config_path, classes_path)

# Uncomment the desired function to test
# process_image('/path/to/image.jpg')
# process_video('/path/to/video.mp4')
process_realtime_video()