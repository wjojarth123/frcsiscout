import cv2
import os
import random
import shutil

class VideoAnnotator:
    def __init__(self, video_path, output_dir, split_ratios=(0.8, 0.1, 0.1)):
        self.cap = cv2.VideoCapture(video_path)
        self.output_dir = output_dir
        self.split_ratios = split_ratios
        self.annotations = []
        self.current_frame = None
        self.current_frame_number = 0
        self.dragging = False
        self.right_dragging = False
        self.drag_start = (0, 0)
        self.drag_end = (0, 0)
        self.class_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]  # 4 classes
        self.class_names = ["Coral", "Algae", "Red", "Blue"]
        self.box_size = 30
        self.recent_bbox_index = -1  # Track most recent bounding box annotation

        # Create directories
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        # Left click handling (Class 2 and 3/4)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_end = (x, y)
            self.show_frame()
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.drag_end = (x, y)
                self.show_frame()
            elif self.right_dragging:
                self.drag_end = (x, y)
                self.show_frame()
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                self.dragging = False
                x1, y1 = self.drag_start
                x2, y2 = self.drag_end
                
                if abs(x2 - x1) > 5 or abs(y2 - y1) > 5:  # Bounding box
                    x_center = (min(x1, x2) + abs(x2 - x1)/2)
                    y_center = (min(y1, y2) + abs(y2 - y1)/2)
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    
                    # Convert to normalized coordinates
                    x_center_n = x_center / self.current_frame.shape[1]
                    y_center_n = y_center / self.current_frame.shape[0]
                    width_n = width / self.current_frame.shape[1]
                    height_n = height / self.current_frame.shape[0]
                    
                    # Add as Class 3 by default
                    self.annotations.append((self.current_frame_number, 2, (x_center_n, y_center_n, width_n, height_n)))
                    self.recent_bbox_index = len(self.annotations) - 1
                    print(f"Class 3 added: {x_center_n:.2f}, {y_center_n:.2f}, {width_n:.2f}x{height_n:.2f}")
                else:  # Class 2 (30x30 box)
                    self.add_fixed_size_box(x, y, 1)
                self.show_frame()

        # Right click handling (Class 1 and copy)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.right_dragging = True
            self.drag_start = (x, y)
            self.drag_end = (x, y)
            self.show_frame()
        elif event == cv2.EVENT_RBUTTONUP:
            if self.right_dragging:
                self.right_dragging = False
                x1, y1 = self.drag_start
                x2, y2 = self.drag_end
                
                if abs(x2 - x1) > 5 or abs(y2 - y1) > 5:  # Copy annotations
                    self.copy_annotations((x1, y1), (x2, y2))
                else:  # Class 1 (30x30 box)
                    self.add_fixed_size_box(x, y, 0)
                self.show_frame()

    # ... (keep add_fixed_size_box, copy_annotations, process_video same as before) ...

    def show_frame(self):
        display_frame = self.current_frame.copy()
        
        # Draw existing annotations
        for ann in self.annotations:
            if ann[0] == self.current_frame_number:
                class_id = ann[1]
                xc, yc, w, h = ann[2]
                
                # Convert to pixel coordinates
                x = int((xc - w/2) * display_frame.shape[1])
                y = int((yc - h/2) * display_frame.shape[0])
                width = int(w * display_frame.shape[1])
                height = int(h * display_frame.shape[0])
                
                color = self.class_colors[class_id]
                cv2.rectangle(display_frame, (x, y), (x+width, y+height), color, 2)
                cv2.putText(display_frame, str(class_id+1), (x+5, y+20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw drag previews and help text
        if self.dragging:
            cv2.rectangle(display_frame, self.drag_start, self.drag_end, self.class_colors[2], 2)
        if self.right_dragging:
            cv2.rectangle(display_frame, self.drag_start, self.drag_end, (255, 255, 0), 2)

        # Updated help text
        help_text = [
            "Left click: Class 2 (30x30)",
            "Right click: Class 1 (30x30)",
            "Left drag: BBox (Class3 - Press R for Class4)",
            "Right drag: Copy from previous",
            "D: Next Frame | A: Previous",
            "Ctrl+Z: Undo | Q: Quit | R: Change to Class4"
        ]
        for i, text in enumerate(help_text):
            y_offset = 30 + 30 * i
            cv2.putText(display_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
            cv2.putText(display_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Frame", display_frame)
    def undo_last_annotation(self):
        """Remove most recent annotation for current frame"""
        for i in reversed(range(len(self.annotations))):
            if self.annotations[i][0] == self.current_frame_number:
                del self.annotations[i]
                print("Undid last annotation")
                self.show_frame()
                break
    def split_dataset(self):
        # Create split directories
        splits = ["train", "val", "test"]
        for split in splits:
            os.makedirs(os.path.join(self.output_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "labels", split), exist_ok=True)

        # Get all valid image-label pairs
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith(".jpg")]
        valid_pairs = []
        
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            label_file = os.path.join(self.labels_dir, f"{base_name}.txt")
            if os.path.exists(label_file):
                valid_pairs.append((img_file, label_file))

        # Shuffle and split
        random.shuffle(valid_pairs)
        total = len(valid_pairs)
        train_end = int(total * self.split_ratios[0])
        val_end = train_end + int(total * self.split_ratios[1])

        # Move files to appropriate directories
        for i, (img_file, label_file) in enumerate(valid_pairs):
            if i < train_end:
                split = "train"
            elif i < val_end:
                split = "val"
            else:
                split = "test"

            # Move image
            src_img = os.path.join(self.images_dir, img_file)
            dst_img = os.path.join(self.output_dir, "images", split, img_file)
            shutil.move(src_img, dst_img)

            # Move label
            src_lbl = label_file
            dst_lbl = os.path.join(self.output_dir, "labels", split, os.path.basename(label_file))
            shutil.move(src_lbl, dst_lbl)

        # Clean up empty directories
        try:
            os.rmdir(self.images_dir)
            os.rmdir(self.labels_dir)
        except OSError:
            pass

        print(f"Dataset split complete: {train_end} train, {val_end-train_end} val, {len(valid_pairs)-val_end} test")
    def process_video(self):
        frame_count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if frame_count % 30 == 0:
                self.current_frame = frame.copy()
                self.current_frame_number = frame_count
                self.recent_bbox_index = -1  # Reset for new frame
                
                # Save frame image
                img_path = os.path.join(self.images_dir, f"frame_{self.current_frame_number:06d}.jpg")
                cv2.imwrite(img_path, self.current_frame)
                
                self.show_frame()
                
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.cap.release()
                        cv2.destroyAllWindows()
                        self.split_dataset()
                        return
                    elif key == ord('d'):
                        self.save_annotations()
                        break
                    elif key == ord('a'):
                        new_frame = max(0, frame_count - 30)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                        frame_count = new_frame
                        break
                    elif key == 26:  # Ctrl+Z
                        self.undo_last_annotation()
                    elif key == ord('r'):
                        self.convert_to_class4()

            frame_count += 1

        self.cap.release()
        cv2.destroyAllWindows()
        self.split_dataset()
    def add_fixed_size_box(self, x, y, class_id):
        """Add a 30x30 pixel box centered at (x,y)"""
        frame_width = self.current_frame.shape[1]
        frame_height = self.current_frame.shape[0]
        x_center = x / frame_width
        y_center = y / frame_height
        width = self.box_size / frame_width
        height = self.box_size / frame_height
        self.annotations.append((self.current_frame_number, class_id, (x_center, y_center, width, height)))
        print(f"Class {class_id+1} added at ({x}, {y})")

    def copy_annotations(self, start_point, end_point):
        """Copy annotations from previous frame within selected area"""
        prev_frame = self.current_frame_number - 30
        if prev_frame < 0:
            print("No previous frame available")
            return
            
        # Convert current selection to normalized coordinates
        frame_width = self.current_frame.shape[1]
        frame_height = self.current_frame.shape[0]
        x1, y1 = start_point
        x2, y2 = end_point
        nx1 = min(x1, x2) / frame_width
        ny1 = min(y1, y2) / frame_height
        nx2 = max(x1, x2) / frame_width
        ny2 = max(y1, y2) / frame_height

        # Find matching annotations from previous frame
        copied = 0
        for ann in self.annotations:
            if ann[0] == prev_frame:
                class_id = ann[1]
                xc, yc, w, h = ann[2]
                # Check if center is within selection
                if nx1 <= xc <= nx2 and ny1 <= yc <= ny2:
                    self.annotations.append((self.current_frame_number, class_id, (xc, yc, w, h)))
                    copied += 1
        print(f"Copied {copied} annotations from frame {prev_frame}")


    def convert_to_class4(self):
        """Convert most recent Class 3 bounding box to Class 4"""
        if self.recent_bbox_index != -1 and self.recent_bbox_index < len(self.annotations):
            frame_num, class_id, coords = self.annotations[self.recent_bbox_index]
            if class_id == 2 and frame_num == self.current_frame_number:
                self.annotations[self.recent_bbox_index] = (frame_num, 3, coords)
                print("Changed to Class 4")
                self.show_frame()
        self.recent_bbox_index = -1
    def save_annotations(self):
        frame_annotations = [ann for ann in self.annotations if ann[0] == self.current_frame_number]
        if not frame_annotations:
            return
            
        yolo_lines = []
        for ann in frame_annotations:
            class_id = ann[1]
            xc, yc, w, h = ann[2]
            yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        
        filename = os.path.join(self.labels_dir, f"frame_{self.current_frame_number:06d}.txt")
        with open(filename, "w") as f:
            f.write("\n".join(yolo_lines))
        print(f"Saved {len(yolo_lines)} annotations to {filename}")

if __name__ == "__main__":
    annotator = VideoAnnotator("final1.mp4", "dataset")
    annotator.process_video()