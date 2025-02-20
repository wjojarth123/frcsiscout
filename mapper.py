import cv2
import numpy as np
import re
import json
from collections import defaultdict

# Configuration
MAP_IMAGE_FILE = 'fieldmap.jpeg'
VIDEO_FILE = 'final1.mp4'
FPS = 30
COLOR_SCHEMES = [
    {'blue': (255, 0, 0), 'red': (0, 0, 255)},     # View 1: Blue/Red
    {'blue': (0, 255, 0), 'red': (0, 165, 255)},   # View 2: Green/Orange
    {'blue': (128, 0, 128), 'red': (0, 255, 255)}, # View 3: Purple/Yellow
]
HOMOGRAPHY_FILE = 'homographies.json'
MERGE_THRESHOLD = 35  # For map point merging
IOU_THRESHOLD = 0.3   # For video box merging

# State constants
IDLE = 0
COLLECTING_VIDEO_POINTS = 1
COLLECTING_MAP_POINTS = 2

# Global variables
current_state = IDLE
video_points = []
map_points = []
calibration_views = []
show_quads = True
dragging = False
selected_view = -1
selected_point_type = None
selected_point_idx = -1
paused = False

def extract_numbers(input_string):
    """Extracts four floating-point numbers from the input string."""
    pattern = r"[-+]?\d*\.\d+"
    matches = re.findall(pattern, input_string)
    return [float(match) for match in matches[:4]]

def parse_tracking_data(data):
    """Parse tracking data with bounding box information"""
    frames = defaultdict(list)
    for line in data.strip().split('\n'):
        parts = line.split(': ')
        frame_num = int(parts[0].split()[1])
        color = parts[1].split(',')[0]
        bbpoints = extract_numbers(parts[1])
        center_x = (bbpoints[0] + bbpoints[2]) / 2
        center_y = ((bbpoints[1] + bbpoints[3]*3) / 4)  # Weighted towards bottom
        if abs(bbpoints[0] - bbpoints[2])>40 and abs(bbpoints[1] - bbpoints[3])>40:
            frames[frame_num].append({
                'color': color,
                'bbox': bbpoints,
                'point': (center_x, center_y)
            })
    return frames

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    # Convert to [x1, y1, x2, y2] format
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    
    # Calculate intersection area
    intersection_x1 = max(b1_x1, b2_x1)
    intersection_y1 = max(b1_y1, b2_y1)
    intersection_x2 = min(b1_x2, b2_x2)
    intersection_y2 = min(b1_y2, b2_y2)
    
    if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
        return 0.0
    
    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
    
    # Calculate union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def merge_video_boxes(frame_data, iou_threshold=IOU_THRESHOLD):
    """Merge overlapping bounding boxes regardless of color"""
    if not frame_data:
        return []
    
    # Sort by area (largest first) to prioritize larger detections
    frame_data = sorted(frame_data, 
                        key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]), 
                        reverse=True)
    
    merged = []
    used = [False] * len(frame_data)
    
    for i in range(len(frame_data)):
        if used[i]:
            continue
        
        current_box = frame_data[i]['bbox']
        current_color = frame_data[i]['color']
        current_point = frame_data[i]['point']
        
        # This object becomes the representative of its group
        merged_obj = {
            'color': current_color,
            'bbox': current_box,
            'point': current_point
        }
        
        used[i] = True
        
        # Check for overlapping boxes
        for j in range(i+1, len(frame_data)):
            if used[j]:
                continue
            
            iou = calculate_iou(current_box, frame_data[j]['bbox'])
            if iou > iou_threshold:
                # Mark as used but keep the original object's properties
                # This gives priority to the largest detection (sorted earlier)
                used[j] = True
        
        merged.append(merged_obj)
    
    return merged

def save_homographies(filename, views):
    """Save homography data to a JSON file"""
    serialized_views = []
    for view in views:
        serialized_view = {
            'src_points': view['src_points'].tolist(),
            'dst_points': view['dst_points'].tolist(),
            'homography': view['homography'].tolist(),
            'active': view['active']
        }
        serialized_views.append(serialized_view)
    
    with open(filename, 'w') as file:
        json.dump(serialized_views, file)
    print(f"Saved {len(views)} homographies to {filename}")

def load_homographies(filename):
    """Load homography data from a JSON file"""
    try:
        with open(filename, 'r') as file:
            serialized_views = json.load(file)
        
        views = []
        for view in serialized_views:
            deserialized_view = {
                'src_points': np.array(view['src_points'], dtype=np.int32),
                'dst_points': np.array(view['dst_points'], dtype=np.int32),
                'homography': np.array(view['homography'], dtype=np.float32),
                'active': view.get('active', True)
            }
            views.append(deserialized_view)
        
        print(f"Loaded {len(views)} homographies from {filename}")
        return views
    except FileNotFoundError:
        print(f"Warning: Homography file {filename} not found. Starting with empty calibration.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not parse homography file {filename}. Starting with empty calibration.")
        return []
    except KeyError:
        print(f"Error: Old homography format detected. Please recalibrate.")
        return []

def transform_point(point, calibration_views):
    for view in calibration_views:
        if cv2.pointPolygonTest(view['src_points'], point, False) >= 0:
            src_pt = np.array([[point]], dtype=np.float32)
            dst_pt = cv2.perspectiveTransform(src_pt, view['homography'])
            return tuple(map(int, dst_pt[0][0]))
    return None
def export_map_positions(tracking_data, calibration_views, output_file="map_positions.json"):
    """Export all mapped positions to a JSON file using all active homographies with merging."""
    print(f"Exporting positions with all homographies and both merges active...")
    
    # Enable both merging features for export
    use_video_merging = True
    use_map_merging = True
    
    # Prepare the export data structure
    export_data = {}
    
    # Process each frame
    for frame_num in sorted(tracking_data.keys()):
        frame_data = tracking_data[frame_num]
        
        # Apply video box merging
        if use_video_merging:
            frame_data = merge_video_boxes(frame_data, IOU_THRESHOLD)
        
        # Apply map merging
        if use_map_merging:
            map_objects = merge_objects(frame_data, calibration_views)
        else:
            map_objects = frame_data
        
        # Store the mapped positions for this frame
        frame_positions = {}
        
        # Process each object in the frame
        for obj in map_objects:
            point = obj['point']
            color = obj['color']
            
            # Try all homographies for each point
            for view_idx, view in enumerate(calibration_views):
                if not view['active']:
                    continue
                    
                if cv2.pointPolygonTest(view['src_points'], point, False) >= 0:
                    src_pt = np.array([[point]], dtype=np.float32)
                    dst_pt = cv2.perspectiveTransform(src_pt, view['homography'])
                    map_point = tuple(map(int, dst_pt[0][0]))
                    
                    # Add to frame positions
                    if color not in frame_positions:
                        frame_positions[color] = []
                    frame_positions[color].append({
                        "x": map_point[0],
                        "y": map_point[1],
                        "view": view_idx
                    })
                    break  # Use first valid homography
        
        # Add frame data to export
        if frame_positions:
            export_data[str(frame_num)] = frame_positions
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Successfully exported {len(export_data)} frames to {output_file}")
    return export_data
def merge_objects(frame_data, calibration_views, threshold=MERGE_THRESHOLD):
    merged_objects = []
    color_groups = defaultdict(list)
    
    for obj in frame_data:
        point = obj['point']
        for view_idx, view in enumerate(calibration_views):
            if cv2.pointPolygonTest(view['src_points'], point, False) >= 0:
                src_pt = np.array([[point]], dtype=np.float32)
                dst_pt = cv2.perspectiveTransform(src_pt, view['homography'])
                map_point = tuple(map(int, dst_pt[0][0]))
                color_groups[obj['color']].append((map_point, view_idx, obj))
                break
    
    for color, objects in color_groups.items():
        merged = []
        for obj in objects:
            found = False
            for group in merged:
                if any(np.linalg.norm(np.array(obj[0]) - np.array(g[0])) < threshold for g in group):
                    group.append(obj)
                    found = True
                    break
            if not found:
                merged.append([obj])
        
        for group in merged:
            best_obj = min(group, key=lambda x: x[1])
            merged_objects.append(best_obj[2])
    
    return merged_objects

def mouse_callback(event, x, y, flags, param):
    global current_state, video_points, map_points, video_width, map_image_resized
    global calibration_views, dragging, selected_view, selected_point_type, selected_point_idx

    if current_state == IDLE:
        # Existing drag handling logic
        if event == cv2.EVENT_LBUTTONDOWN:
            min_dist = 10
            closest = None
            map_x = x - video_width

            if x < video_width:  # Video area
                for v_idx, view in enumerate(calibration_views):
                    for p_idx, (px, py) in enumerate(view['src_points']):
                        dist = np.hypot(x-px, y-py)
                        if dist < min_dist:
                            closest = (v_idx, 'src', p_idx)
                            min_dist = dist
            elif 0 <= map_x < map_image_resized.shape[1]:  # Map area
                for v_idx, view in enumerate(calibration_views):
                    for p_idx, (px, py) in enumerate(view['dst_points']):
                        dist = np.hypot(map_x-px, y-py)
                        if dist < min_dist:
                            closest = (v_idx, 'dst', p_idx)
                            min_dist = dist
            
            if closest:
                dragging = True
                selected_view, selected_point_type, selected_point_idx = closest

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            if selected_point_type == 'src' and x < video_width:
                calibration_views[selected_view]['src_points'][selected_point_idx] = (x, y)
            elif selected_point_type == 'dst':
                map_x = x - video_width
                if 0 <= map_x < map_image_resized.shape[1]:
                    calibration_views[selected_view]['dst_points'][selected_point_idx] = (map_x, y)

        elif event == cv2.EVENT_LBUTTONUP and dragging:
            view = calibration_views[selected_view]
            src_pts = np.array(view['src_points'], dtype=np.float32).reshape(-1,1,2)
            dst_pts = np.array(view['dst_points'], dtype=np.float32).reshape(-1,1,2)
            view['homography'], _ = cv2.findHomography(src_pts, dst_pts)
            dragging = False
            selected_view = -1

    else:  # Calibration point collection mode
        if event == cv2.EVENT_LBUTTONDOWN:
            if current_state == COLLECTING_VIDEO_POINTS and x < video_width:
                video_points.append((x, y))
                if len(video_points) == 4:
                    current_state = COLLECTING_MAP_POINTS
            elif current_state == COLLECTING_MAP_POINTS and x >= video_width:
                map_x = x - video_width
                if map_x < map_image_resized.shape[1]:
                    map_points.append((map_x, y))
                    if len(map_points) == 4:
                        current_state = IDLE

# Load tracking data
with open('positions.txt', 'r') as file:
    input_data = file.read()
tracking_data = parse_tracking_data(input_data)

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_FILE)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load map image and resize
map_image = cv2.imread(MAP_IMAGE_FILE)
map_aspect_ratio = map_image.shape[1] / map_image.shape[0]
map_image_resized = cv2.resize(map_image, (int(video_height * map_aspect_ratio), video_height))

cv2.namedWindow('Tracking Visualization')
cv2.setMouseCallback('Tracking Visualization', mouse_callback)
frame_nums = sorted(tracking_data.keys())
current_frame_idx = 0
enable_merging = False
enable_video_merging = False

# Try to load existing homographies
try:
    #calibration_views = load_homographies(HOMOGRAPHY_FILE)
    pass
except Exception as e:
    print(f"Error loading homographies: {e}")
    calibration_views = []

while True:
    if current_frame_idx >= len(frame_nums):
        current_frame_idx = 0  # Loop back to start
    
    frame_num = frame_nums[current_frame_idx]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, video_frame = cap.read()
    if not ret:
        break

    frame_data = tracking_data.get(frame_num, [])
    
    # Apply video box merging if enabled
    if enable_video_merging:
        frame_data = merge_video_boxes(frame_data, IOU_THRESHOLD)
    
    # Apply map point merging if enabled
    if enable_merging:
        map_objects = merge_objects(frame_data, calibration_views)
    else:
        map_objects = frame_data

    # Draw video objects
    for obj in frame_data:
        color = COLOR_SCHEMES[0][obj['color']]  # Original colors for video
        center = tuple(map(int, obj['point']))
        
        # Draw bounding box if available
        if 'bbox' in obj:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            cv2.rectangle(video_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw center point
        cv2.circle(video_frame, center, 5, color, -1)

    # Draw map projections for active views
    map_display = map_image_resized.copy()
    for view_idx, view in enumerate(calibration_views):
        if not view['active']:
            continue
            
        color_scheme = COLOR_SCHEMES[view_idx % len(COLOR_SCHEMES)]
        for obj in map_objects:
            point = obj['point']
            if cv2.pointPolygonTest(view['src_points'], point, False) >= 0:
                src_pt = np.array([[point]], dtype=np.float32)
                dst_pt = cv2.perspectiveTransform(src_pt, view['homography'])
                map_point = tuple(map(int, dst_pt[0][0]))
                cv2.circle(map_display, map_point, 5, color_scheme[obj['color']], -1)

    # Draw UI elements and status text
    status_lines = [
        f"Frame: {frame_num}",
        f"Video Merging: {'ON' if enable_video_merging else 'OFF'} (V)",
        f"Map Merging: {'ON' if enable_merging else 'OFF'} (M)",
        f"Playback: {'PAUSED' if paused else 'PLAYING'} (P)",
        "Press 'N' for new calibration"
        "Press 'E' to export map positions"
    ]
    
    for i, text in enumerate(status_lines):
        cv2.putText(map_display, text, (10, 30 + i*25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    
    # Draw view activation status
    y_offset = 30 + len(status_lines)*25
    for idx, view in enumerate(calibration_views[:3]):
        y_pos = y_offset + idx*25
        color = COLOR_SCHEMES[idx % len(COLOR_SCHEMES)]['blue']
        status = f"View {idx+1}: {'ON' if view['active'] else 'OFF'} (Press {idx+1})"
        cv2.putText(map_display, status, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw calibration quads if enabled
    if show_quads:
        for view in calibration_views:
            if not view['active']:
                continue
            src_quad = view['src_points']
            for i in range(4):
                cv2.line(video_frame, tuple(src_quad[i]), tuple(src_quad[(i+1)%4]), (0,255,0), 2)
            dst_quad = view['dst_points']
            for i in range(4):
                cv2.line(map_display, tuple(dst_quad[i]), tuple(dst_quad[(i+1)%4]), (0,255,0), 2)

    combined = np.hstack((video_frame, map_display))
    cv2.imshow('Tracking Visualization', combined)

    # Handle input
    wait_time = 1 if paused else int(1000/FPS)
    key = cv2.waitKey(wait_time) & 0xFF
    
    if key == ord('n'):
        # Start new calibration - pause video and show static frame
        paused = True  # Auto-pause when starting calibration
        current_state = COLLECTING_VIDEO_POINTS
        video_points = []
        map_points = []
        cal_video = video_frame.copy()
        cal_map = map_image_resized.copy()

        while True:
            # Draw video points
            temp_vid = cal_video.copy()
            for p in video_points:
                cv2.circle(temp_vid, p, 5, (0,255,0), -1)
            if len(video_points) > 1:
                for i in range(len(video_points)-1):
                    cv2.line(temp_vid, video_points[i], video_points[i+1], (0,255,0), 2)
                if len(video_points) == 4:
                    cv2.line(temp_vid, video_points[3], video_points[0], (0,255,0), 2)

            # Draw map points
            temp_map = cal_map.copy()
            for p in map_points:
                cv2.circle(temp_map, p, 5, (0,255,0), -1)
            if len(map_points) > 1:
                for i in range(len(map_points)-1):
                    cv2.line(temp_map, map_points[i], map_points[i+1], (0,255,0), 2)
                if len(map_points) == 4:
                    cv2.line(temp_map, map_points[3], map_points[0], (0,255,0), 2)

            # Display current calibration instructions
            instruction = "Click to place points on video" if current_state == COLLECTING_VIDEO_POINTS else "Click to place points on map"
            points_needed = 4 - len(video_points) if current_state == COLLECTING_VIDEO_POINTS else 4 - len(map_points)
            cv2.putText(temp_map, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            cv2.putText(temp_map, f"Points needed: {points_needed}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            cv2.putText(temp_map, "ESC to cancel", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

            combined_cal = np.hstack((temp_vid, temp_map))
            cv2.imshow('Tracking Visualization', combined_cal)
            
            k = cv2.waitKey(1)
            if k == 27:  # ESC to cancel calibration
                current_state = IDLE
                break
            elif current_state == IDLE:
                if len(video_points) == 4 and len(map_points) == 4:
                    src_pts = np.array(video_points, dtype=np.float32).reshape(-1,1,2)
                    dst_pts = np.array(map_points, dtype=np.float32).reshape(-1,1,2)
                    homography, _ = cv2.findHomography(src_pts, dst_pts)
                    calibration_views.append({
                        'src_points': np.array(video_points, dtype=np.int32),
                        'dst_points': np.array(map_points, dtype=np.int32),
                        'homography': homography,
                        'active': True
                    })
                break
    elif key == ord('1') and len(calibration_views) > 0:
        calibration_views[0]['active'] = not calibration_views[0]['active']
    elif key == ord('2') and len(calibration_views) > 1:
        calibration_views[1]['active'] = not calibration_views[1]['active']
    elif key == ord('3') and len(calibration_views) > 2:
        calibration_views[2]['active'] = not calibration_views[2]['active']
    elif key == ord('g'):
        show_quads = not show_quads
    elif key == ord('s'):
        save_homographies(HOMOGRAPHY_FILE, calibration_views)
    elif key == ord('l'):
        calibration_views = load_homographies(HOMOGRAPHY_FILE)
    elif key == ord('m'):
        enable_merging = not enable_merging
    elif key == ord('v'):
        enable_video_merging = not enable_video_merging
    elif key == ord('p'):
        paused = not paused
    elif key == ord('e'):
        # Export map positions using all homographies with both merges active
        export_file = "map_positions.json"
        export_map_positions(tracking_data, calibration_views, export_file)
        # Show export confirmation on screen
        export_message = f"Exported positions to {export_file}"
        cv2.putText(map_display, export_message, (10, map_display.shape[0]-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    elif key == ord(',') and paused:  # Previous frame when paused
        current_frame_idx = max(0, current_frame_idx - 1)
    elif key == ord('.') and paused:  # Next frame when paused
        current_frame_idx = min(len(frame_nums) - 1, current_frame_idx + 1)
    elif key == 27:  # ESC
        break
    
    # Advance to next frame if not paused
    if not paused:
        current_frame_idx += 1

cap.release()
cv2.destroyAllWindows()