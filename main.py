import cv2
import numpy as np
from object_tracker import ObjectTracker

# Path to your YOLOv8 model
model_path = "/home/tg0006/Desktop/footfall/yolo11x.pt"  # Ensure the correct path to your YOLO model

# Path to your input video
input_video_path = "/home/tg0006/Desktop/footfall/video1.mp4"  # Change this to your video path

# Path for the output video
output_video_path = "/home/tg0006/Desktop/footfall/output_video.mp4"  # Change this to your desired output path

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initialize object tracker
tracker = ObjectTracker(model_path)

# Mouse callback function to capture points
line_points = []

def select_point(event, x, y, flags, param):
    global line_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
            if len(line_points) == 2:
                tracker.update_line(line_points)  # Update the tracker with the drawn line

# Create a named window and set the mouse callback
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', select_point)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the line if points are selected
    if tracker.line_drawn:
        cv2.line(frame, line_points[0], line_points[1], (0, 255, 0), 2)

    # Perform tracking
    frame = tracker.track_objects(frame)

    # Display the crossing count
    cv2.putText(frame, f"Crossing Count: {tracker.get_count()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Write the frame to the output video
    out.write(frame)

    # Show frame
    cv2.imshow('Video', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to: {output_video_path}")