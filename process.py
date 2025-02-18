import cv2
import os
import json

usesaved=True

def get_frame_at_time(cap, seconds):
    """
    Jump to a specific time (in seconds) in the video and return the frame.
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ret, frame = cap.read()
    if not ret:
        print(f"Could not retrieve frame at {seconds} seconds.")
        return None
    return frame

def capture_rois(frame, num_rois, window_name="Select ROIs"):
    """
    Let the user select multiple ROIs on a given frame.
    The function uses OpenCV's built-in selectROIs.
    """
    # OpenCV's selectROIs returns a numpy array of selected ROIs.
    rois = cv2.selectROIs(window_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    if len(rois) != num_rois:
        print(f"Warning: Expected {num_rois} ROIs, but got {len(rois)}. You may re-run to select correctly.")
    return rois.tolist()  # Convert numpy array to list

def save_rois(roi_data, filename="roi_positions.json"):
    """
    Save the ROI data to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(roi_data, f, indent=4)
    print(f"ROI data saved to {filename}.")

def load_rois(filename="roi_positions.json"):
    """
    Load the ROI data from a JSON file.
    """
    with open(filename, "r") as f:
        roi_data = json.load(f)
    print(f"Loaded ROI data from {filename}.")
    return roi_data

def process_video(video_path, roi_data):
    """
    Process the video using the predefined ROIs for each segment.
    Splits the video into frames and then crops out the camera view regions.
    """
    # Create an output folder named after the video (without extension)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_base = video_name
    os.makedirs(output_base, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    # Define the segments (in seconds)
    # Segment 1: From 8 s to 2:22 (142 s) using layout1 (3 views)
    seg1_start, seg1_end = 8, 142
    # Segment 2: From 2:22 (142 s) to 2:48 (168 s) using layout2 (4 views)
    seg2_start, seg2_end = 142, 168

    frame_idx = 0  # We'll number the output frame folders consecutively

    # Process Segment 1 (Layout1)
    print("Processing Segment 1 (8 s to 2:22) with 3 camera views...")
    cap.set(cv2.CAP_PROP_POS_MSEC, seg1_start * 1000)
    while True:
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_time > seg1_end:
            break
            
        ret, frame = cap.read()
        if not ret:
            break

        # Create folder for this frame
        frame_folder = os.path.join(output_base, f"frame_{frame_idx:06d}")
        os.makedirs(frame_folder, exist_ok=True)

        # Crop each ROI for layout1
        for idx, roi in enumerate(roi_data["layout1"]):
            x, y, w, h = roi
            crop = frame[y:y+h, x:x+w]
            out_path = os.path.join(frame_folder, f"camera_view_{idx+1}.jpg")
            cv2.imwrite(out_path, crop)
        frame_idx += 1

    # Process Segment 2 (Layout2)
    print("Processing Segment 2 (2:22 to 2:48) with 4 camera views...")
    cap.set(cv2.CAP_PROP_POS_MSEC, seg2_start * 1000)
    while True:
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_time > seg2_end:
            break
        ret, frame = cap.read()
        if not ret:
            break

        # Create folder for this frame
        frame_folder = os.path.join(output_base, f"frame_{frame_idx:06d}")
        os.makedirs(frame_folder, exist_ok=True)

        # Crop each ROI for layout2
        for idx, roi in enumerate(roi_data["layout2"]):
            x, y, w, h = roi
            crop = frame[y:y+h, x:x+w]
            out_path = os.path.join(frame_folder, f"camera_view_{idx+1}.jpg")
            cv2.imwrite(out_path, crop)
        frame_idx += 1

    cap.release()
    print("Video processing complete!")

if __name__ == "__main__":

    # Open the video
    cap = cv2.VideoCapture("final2.mp4")
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(1)

    roi_data = {}
    if usesaved and os.path.exists("roi_positions.json"):
        roi_data = load_rois("roi_positions.json")
    else:
        # --- Capture ROIs manually ---
        # Layout 1: from frame at 10 s (expecting 3 camera views)
        frame_10s = get_frame_at_time(cap, 10)
        if frame_10s is None:
            print("Failed to capture frame at 10 seconds.")
            exit(1)
        print("Select 3 ROIs for Layout 1 (from frame at 10 s).")
        rois_layout1 = capture_rois(frame_10s, 3, window_name="Select ROIs for Layout 1")
        roi_data["layout1"] = rois_layout1

        # Layout 2: from frame at 2:30 (150 s) (expecting 4 camera views)
        frame_230 = get_frame_at_time(cap, 150)
        if frame_230 is None:
            print("Failed to capture frame at 2:30 seconds.")
            exit(1)
        print("Select 4 ROIs for Layout 2 (from frame at 2:30).")
        rois_layout2 = capture_rois(frame_230, 4, window_name="Select ROIs for Layout 2")
        roi_data["layout2"] = rois_layout2

        # Save the ROI data to file
        save_rois(roi_data, "roi_positions.json")

    cap.release()

    # Process the video to split frames and save the camera views.
    process_video("final2.mp4", roi_data)
