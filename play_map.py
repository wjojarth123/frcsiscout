import json
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# ---------------------------
# EKF (Kalman Filter) Tracker with Persistent Snapping and Velocity Decay
# ---------------------------
class KalmanTracker:
    def __init__(self, init_x, init_y, dt=1.0):
        self.dt = dt
        # State vector: [x, y, vx, vy]
        self.x = np.array([init_x, init_y, 0, 0], dtype=float)
        # Initial covariance (tuned lower for faster correction)
        self.P = np.eye(4) * 50.0

        # State transition model: constant velocity
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]])

        # Measurement matrix: we only measure position [x, y]
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Reduced measurement noise covariance to trust the measurements more
        self.R = np.eye(2) * 1.0

        # Process noise covariance (tuned lower to allow snapping)
        self.Q = np.eye(4) * 0.1

        # Counter for consecutive high innovations
        self.snap_counter = 0
        # Number of consecutive frames with high innovation required to snap
        self.snap_required_count = 20
        # Velocity decay factor (between 0 and 1)
        self.velocity_decay = 0.2

    def predict(self):
        # Predict state and covariance
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Decay the velocity component so that old velocity doesn't overly influence prediction
        self.x[2:4] *= self.velocity_decay
        return self.x.copy()

    def update(self, z, snap_threshold=30):
        """
        Update the state with measurement z.
        If the innovation (difference) is large for snap_required_count consecutive frames,
        then snap directly to the measurement.
        """
        y = z - (self.H @ self.x)
        error = np.linalg.norm(y)
        
        if error > snap_threshold:
            self.snap_counter += 1
            if self.snap_counter >= self.snap_required_count:
                # Snap to measurement since the error has been persistently high.
                self.x[0:2] = z
                self.x[2:4] = 0  # Reset velocity
                self.P = np.eye(4) * 50.0  # Reset covariance
                self.snap_counter = 0
                return self.x.copy()
        else:
            # Innovation is acceptable; reset the counter.
            self.snap_counter = 0

        # Standard Kalman update
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy()

# ------------------------------------------------------
# Function to assign detections to trackers (Hungarian Algorithm)
# ------------------------------------------------------
def assign_detections_to_trackers(trackers, detections, threshold=500):
    """
    Given a list of tracker objects and a list of detection points ([x, y]),
    assign detections to trackers using the Hungarian algorithm.
    We use a higher threshold to force assignment even if detections are farther from the predicted position.
    """
    if len(detections) == 0:
        return []  # no detections to assign

    cost_matrix = np.zeros((len(trackers), len(detections)))
    for t, tracker in enumerate(trackers):
        pred = tracker.x[:2]
        for d, det in enumerate(detections):
            cost_matrix[t, d] = np.linalg.norm(pred - det)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < threshold:
            assignments.append((r, c))
    return assignments

# ------------------------------------------------------
# Helper: Lighten a BGR color by blending it with white.
# ------------------------------------------------------
def lighten_color(color, factor=0.5):
    return tuple(int(c + (255 - c) * factor) for c in color)

# ------------------------------------------------------
# Helper: Combine map and video frames side-by-side.
# ------------------------------------------------------
def combine_frames(map_img, video_frame):
    # Resize video_frame to match the height of map_img
    map_h = map_img.shape[0]
    vid_h, vid_w = video_frame.shape[:2]
    scale = map_h / vid_h
    new_w = int(vid_w * scale)
    video_resized = cv2.resize(video_frame, (new_w, map_h))
    # Stack map and video side by side
    combined = np.hstack((map_img, video_resized))
    return combined

# ---------------------------
# Main function
# ---------------------------
def main():
    # Load JSON data (ensure your JSON file is named 'map_positions.json' and is in the same folder)
    with open('map_positions.json', 'r') as f:
        data = json.load(f)

    # The keys in the JSON represent frame numbers; sort them numerically.
    frame_numbers = sorted([int(key) for key in data.keys()])

    # Initialize trackers for each team; each team should always have 3 trackers
    trackers = {'red': [], 'blue': []}
    first_frame = data[str(frame_numbers[0])]
    default_pos = (320, 240)  # fallback initial position if no detection is available

    for team in ['red', 'blue']:
        team_detections = first_frame.get(team, [])
        for i in range(3):
            if i < len(team_detections):
                det = team_detections[i]
                tracker = KalmanTracker(det['x'], det['y'])
            else:
                tracker = KalmanTracker(default_pos[0], default_pos[1])
            trackers[team].append(tracker)

    # Open video file (assumed to be "video.mp4" in the same folder)
    video_cap = cv2.VideoCapture('final1.mp4')
    if not video_cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Create a display window using OpenCV
    cv2.namedWindow("Tracking and Video", cv2.WINDOW_NORMAL)

    # Define colors (BGR) for filtered positions
    team_colors = {
        'red': (0, 0, 255),
        'blue': (255, 0, 0)
    }

    # Main loop: iterate over each frame number from the JSON data
    for frame_num in frame_numbers:
        frame_data = data[str(frame_num)]

        # Set the video capture to the current frame number from the JSON
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, video_frame = video_cap.read()
        if not ret:
            print(f"Could not read video frame {frame_num}.")
            break

        # Process each team separately
        for team in ['red', 'blue']:
            detections = []
            for det in frame_data.get(team, []):
                detections.append(np.array([det['x'], det['y']]))
            
            # Predict the state of each tracker
            for tracker in trackers[team]:
                tracker.predict()

            # If there are detections, assign them to trackers and update
            if detections:
                assignments = assign_detections_to_trackers(trackers[team], detections, threshold=500)
                for t_idx, d_idx in assignments:
                    trackers[team][t_idx].update(detections[d_idx])
            # Unassigned trackers continue with their predicted state

        # Create a base map image (white background, scaled up for clarity)
        map_img = np.ones((480*2, 640*2, 3), dtype=np.uint8) * 255

        # ------------------------------
        # Draw raw (unfiltered) detections with lighter circles
        # ------------------------------
        overlay = map_img.copy()
        for team in ['red', 'blue']:
            raw_color = lighten_color(team_colors[team], factor=0.5)
            for det in frame_data.get(team, []):
                x, y = int(det['x']), int(det['y'])
                cv2.circle(overlay, (x, y), 10, raw_color, -1)
        alpha = 0.3
        map_img = cv2.addWeighted(overlay, alpha, map_img, 1 - alpha, 0)

        # ------------------------------
        # Draw filtered (tracked) positions
        # ------------------------------
        for team in ['red', 'blue']:
            for idx, tracker in enumerate(trackers[team]):
                x, y = int(tracker.x[0]), int(tracker.x[1])
                cv2.circle(map_img, (x, y), 10, team_colors[team], -1)
                cv2.putText(map_img, f"{team}-{idx}", (x - 20, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Combine map image with the current video frame side by side
        combined_frame = combine_frames(map_img, video_frame)

        cv2.imshow("Tracking and Video", combined_frame)
        # Display each frame for ~33ms; press ESC to exit early
        if cv2.waitKey(33) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    video_cap.release()

if __name__ == "__main__":
    main()
