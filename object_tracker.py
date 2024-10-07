import cv2
import numpy as np
from ultralytics import YOLO

class ObjectTracker:
    def __init__(self, model_path, line_points=None):
        self.model = YOLO(model_path)
        self.line_points = line_points or []
        self.line_drawn = False
        self.count = 0
        self.track_id_counter = 0
        self.previous_boxes = {}
        self.crossed_ids = set()
        self.debug = True

    def update_line(self, line_points):
        self.line_points = line_points
        self.line_drawn = True
        if self.debug:
            print(f"Line updated: {self.line_points}")

    def get_iou(self, box1, box2):
        # Calculate IoU between two boxes
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def line_intersection(self, line1, line2):
        # Check if two line segments intersect
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]
        
        den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if den == 0:
            return False
        
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
        
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            return True
        return False

    def track_objects(self, frame):
        results = self.model(frame)
        boxes = results[0].boxes

        current_boxes = {}

        if self.line_drawn and len(self.line_points) == 2:
            cv2.line(frame, self.line_points[0], self.line_points[1], (255, 0, 0), 2)

        for box in boxes:
            if box.cls == 0:  # 'person' class
                if box.xyxy is not None and len(box.xyxy) > 0:
                    box_xyxy = box.xyxy.cpu().numpy()[0]
                    x1, y1, x2, y2 = map(int, box_xyxy)

                    # Generate a new ID or match with an existing one
                    matched_id = None
                    for prev_id, prev_box in self.previous_boxes.items():
                        if self.get_iou((x1, y1, x2, y2), prev_box) > 0.5:  # You can adjust this threshold
                            matched_id = prev_id
                            break

                    if matched_id is None:
                        current_id = self.track_id_counter
                        self.track_id_counter += 1
                    else:
                        current_id = matched_id

                    current_boxes[current_id] = (x1, y1, x2, y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if self.debug:
                        print(f"Object {current_id}: ({x1}, {y1}, {x2}, {y2})")

                    if self.line_drawn:
                        if current_id in self.previous_boxes:
                            prev_box = self.previous_boxes[current_id]
                            prev_x1, prev_y1, prev_x2, prev_y2 = prev_box

                            # Check if the line intersects with the path of the bounding box
                            box_path = ((prev_x1, prev_y1), (x1, y1))
                            if self.line_intersection(self.line_points, box_path) and current_id not in self.crossed_ids:
                                self.count += 1
                                self.crossed_ids.add(current_id)
                                if self.debug:
                                    print(f"Object {current_id} crossed the line. Count: {self.count}")

                    self.previous_boxes[current_id] = (x1, y1, x2, y2)

        # Clean up previous_boxes to remove objects no longer in frame
        self.previous_boxes = {k: v for k, v in self.previous_boxes.items() if k in current_boxes}

        return frame

    def get_count(self):
        return self.count