import numpy as np
from sort import Sort

# Initialize the Sort tracker
tracker = Sort()

# Example YOLO output for each frame: [frame_number, x1, y1, x2, y2, class_id, confidence]
yolo_detections = [
    [0, 100, 150, 200, 250, 0, 0.9],
    [0, 300, 400, 350, 450, 0, 0.85],
    # More detections per frame...
]

# Dictionary to store tracked objects
tracked_objects = {}

# Process each frame
for detection in yolo_detections:
    frame_number, x1, y1, x2, y2, class_id, confidence = detection
    
    # Prepare the detection in the format [x1, y1, x2, y2, score]
    det = np.array([x1, y1, x2, y2, confidence]).reshape(1, -1)
    
    # Update the tracker
    tracked = tracker.update(det)
    
    # Collect tracked object data
    if frame_number not in tracked_objects:
        tracked_objects[frame_number] = []
    
    for obj in tracked:
        tracked_objects[frame_number].append(obj)
    
# Print tracked objects
for frame_number, objects in tracked_objects.items():
    print(f"Frame {frame_number}:")
    for obj in objects:
        obj_id, x1, y1, x2, y2 = obj
        print(f"  Object ID {obj_id}: ({x1}, {y1}, {x2}, {y2})")