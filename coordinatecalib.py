import cv2
import numpy as np
import os

# -------------------------------
# Helper class for a view rectangle
# -------------------------------
class RectView:
    def __init__(self, rect, color):
        """
        rect: (x, y, w, h) in the video frame coordinates.
        color: BGR tuple for drawing.
        """
        self.rect = rect  
        self.correspondences = []  # List of (view_point, map_point) tuples.
        self.H = None              # Homography matrix (once computed).
        self.color = color

    def add_correspondence(self, view_pt, map_pt):
        self.correspondences.append((view_pt, map_pt))

    def compute_homography(self):
        """Compute a homography using all correspondences (if at least 4 exist)."""
        if len(self.correspondences) >= 4:
            src_pts = np.array([pt[0] for pt in self.correspondences], dtype=np.float32)
            dst_pts = np.array([pt[1] for pt in self.correspondences], dtype=np.float32)
            H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            self.H = H
            print("Computed homography:\n", H)
            return H
        else:
            print("Not enough points to compute homography (need at least 4).")
        return None

    def get_projected_rect(self):
        """Project the four corners of the view rectangle using the computed homography."""
        if self.H is not None:
            x, y, w, h = self.rect
            pts = np.array([[x, y],
                            [x + w, y],
                            [x + w, y + h],
                            [x, y + h]], dtype=np.float32).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(pts, self.H)
            return projected.astype(int)
        return None

# -------------------------------
# Global variables and state
# -------------------------------
# Base images (will be loaded later)
frame_img = None  # The video frame (frame 1000)
map_img = None    # The map image (or blank canvas)

# List of view rectangles (instances of RectView)
rect_views = []

# For drawing a new view rectangle on the frame
drawing = False
rect_start = None       # Starting point (x, y) when dragging
current_rect = None     # Current rectangle (x1, y1, x2, y2)

# For adding correspondences:
# When you click inside a view (on the frame) that is NOT a drag,
# we store the view point here until you click on the map.
pending_corr = None  # Dictionary with keys: "view_index", "view_pt"

# For test mode (when 't' is pressed)
test_mode = False
# Markers to show test points
test_markers_frame = []  # List of (x, y, color) for points on the frame
test_markers_map = []    # List of (x, y, color) for points on the map

# -------------------------------
# Mouse callbacks
# -------------------------------

def frame_mouse_callback(event, x, y, flags, param):
    global drawing, rect_start, current_rect, pending_corr, rect_views, test_mode, test_markers_map

    # -------- Test mode: click on frame to show corresponding point on map --------
    if test_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            # For each view that contains this point (and has a computed homography),
            # compute the map location and mark it.
            for view in rect_views:
                vx, vy, vw, vh = view.rect
                if vx <= x <= vx + vw and vy <= y <= vy + vh:
                    if view.H is not None:
                        pts = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
                        mapped = cv2.perspectiveTransform(pts, view.H)
                        mx, my = mapped[0][0]
                        test_markers_map.append((int(mx), int(my), view.color))
                        print(f"Frame click ({x}, {y}) -> Map point ({int(mx)},{int(my)})")
        return  # Do not process further if in test mode

    # -------- Normal (edit) mode: handle drawing new views or adding correspondences --------
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing a new rectangle
        drawing = True
        rect_start = (x, y)
        current_rect = (x, y, x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update current rectangle coordinates as you drag
            current_rect = (rect_start[0], rect_start[1], x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            x1, y1, x2, y2 = current_rect
            # Normalize coordinates so that (x_min,y_min) is top-left.
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)
            # If the drag distance is significant, create a new view rectangle.
            if abs(x_max - x_min) > 10 and abs(y_max - y_min) > 10:
                new_rect = (x_min, y_min, x_max - x_min, y_max - y_min)
                # Choose a random color for this view
                color = (int(np.random.randint(0, 255)),
                         int(np.random.randint(0, 255)),
                         int(np.random.randint(0, 255)))
                new_view = RectView(new_rect, color)
                rect_views.append(new_view)
                print(f"Created new view rectangle: {new_rect}")
            else:
                # Otherwise, treat this as a simple click.
                # Check if the click falls inside any view rectangle.
                for idx, view in enumerate(rect_views):
                    vx, vy, vw, vh = view.rect
                    if vx <= x <= vx + vw and vy <= y <= vy + vh:
                        pending_corr = {"view_index": idx, "view_pt": (x, y)}
                        print(f"Selected view {idx} for correspondence, view point: {(x, y)}")
                        break

def map_mouse_callback(event, x, y, flags, param):
    global pending_corr, rect_views, test_mode, test_markers_frame

    # -------- Test mode: click on map to show corresponding points on the frame --------
    if test_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            for view in rect_views:
                if view.H is not None:
                    try:
                        H_inv = np.linalg.inv(view.H)
                    except np.linalg.LinAlgError:
                        continue
                    pts = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
                    mapped = cv2.perspectiveTransform(pts, H_inv)
                    mx, my = mapped[0][0]
                    test_markers_frame.append((int(mx), int(my), view.color))
                    print(f"Map click ({x}, {y}) -> Frame point ({int(mx)},{int(my)})")
        return

    # -------- Normal (edit) mode: if a view point was selected, add the map correspondence --------
    if event == cv2.EVENT_LBUTTONDOWN:
        if pending_corr is not None:
            view_index = pending_corr["view_index"]
            view_pt = pending_corr["view_pt"]
            map_pt = (x, y)
            rect_views[view_index].add_correspondence(view_pt, map_pt)
            print(f"Added correspondence for view {view_index}: {view_pt} -> {map_pt}")
            pending_corr = None

# -------------------------------
# Function to draw overlays on the images
# -------------------------------
def draw_overlays():
    global frame_img, map_img, current_rect, drawing, rect_views, test_markers_frame, test_markers_map

    # Make copies so we do not alter the originals.
    frame_disp = frame_img.copy()
    map_disp = map_img.copy()

    # Draw each view rectangle on the frame and its correspondence points.
    for view in rect_views:
        x, y, w, h = view.rect
        cv2.rectangle(frame_disp, (x, y), (x + w, y + h), view.color, 2)
        for (vp, mp) in view.correspondences:
            cv2.circle(frame_disp, (int(vp[0]), int(vp[1])), 5, view.color, -1)
        # If homography exists, project the view rectangle onto the map.
        if view.H is not None:
            projected = view.get_projected_rect()
            if projected is not None:
                pts = projected.reshape(-1, 2)
                cv2.polylines(map_disp, [pts], isClosed=True, color=view.color, thickness=2)
                # Also show the map correspondence points.
                for (vp, mp) in view.correspondences:
                    cv2.circle(map_disp, (int(mp[0]), int(mp[1])), 5, view.color, -1)

    # If we are in the middle of drawing a rectangle, show it.
    if drawing and current_rect is not None:
        x1, y1, x2, y2 = current_rect
        cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Draw any test markers (from test mode)
    for (x, y, color) in test_markers_frame:
        cv2.circle(frame_disp, (x, y), 8, color, 2)
    for (x, y, color) in test_markers_map:
        cv2.circle(map_disp, (x, y), 8, color, 2)

    return frame_disp, map_disp

# -------------------------------
# Main function
# -------------------------------
def main():
    global frame_img, map_img, test_mode, rect_views, test_markers_frame, test_markers_map

    # ---- Open video and jump to frame 1000 ----
    video_path = "final1.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file!")
        return

    # Set video position to frame 1000
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame 1000")
        return
    frame_img = frame.copy()

    # ---- Load the map image (or create a blank canvas) ----
    map_path = "fieldmap.jpeg"
    if os.path.exists(map_path):
        map_img = cv2.imread(map_path)
    else:
        # Create a blank map (e.g., 800x600 white canvas)
        map_img = 255 * np.ones((600, 800, 3), dtype=np.uint8)
    # Resize map if needed (here we keep original size)

    # ---- Create windows and set mouse callbacks ----
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", frame_mouse_callback)
    cv2.namedWindow("Map")
    cv2.setMouseCallback("Map", map_mouse_callback)

    print("Instructions:")
    print("  - On the 'Frame' window, click & drag to create a view rectangle.")
    print("  - Then click (a simple click) inside a view to select a point for correspondence,")
    print("    then click on the 'Map' window to add the corresponding point.")
    print("  - Repeat to add at least 4 correspondences per view.")
    print("  - Press 'c' to compute the homography for each view and project its rectangle onto the map.")
    print("  - Press 't' to toggle test mode: clicking in one window shows the corresponding point in the other.")
    print("  - Press 'r' to clear test markers. Press 'q' to quit.")

    while True:
        disp_frame, disp_map = draw_overlays()
        cv2.imshow("Frame", disp_frame)
        cv2.imshow("Map", disp_map)
        key = cv2.waitKey(20) & 0xFF

        # Compute homographies for all views when 'c' is pressed.
        if key == ord('c'):
            for idx, view in enumerate(rect_views):
                print(f"\nView {idx}:")
                view.compute_homography()
        # Toggle test mode with 't'
        elif key == ord('t'):
            test_mode = not test_mode
            mode = "TEST mode" if test_mode else "EDIT mode"
            print("Switched to", mode)
        # Clear test markers with 'r'
        elif key == ord('r'):
            test_markers_frame = []
            test_markers_map = []
            print("Cleared test markers.")
        # Quit with 'q'
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
