import cv2
import numpy as np
import re
import json
from collections import defaultdict

# Configuration
MAP_IMAGE_FILE = 'fieldmap.jpeg'
VIDEO_FILE = 'final2.mp4'
FPS = 30
COLOR_SCHEMES = [
    {'blue': (255, 0, 0), 'red': (0, 0, 255)},     # View 1: Blue/Red
    {'blue': (0, 255, 0), 'red': (0, 165, 255)},   # View 2: Green/Orange
    {'blue': (128, 0, 128), 'red': (0, 255, 255)}, # View 3: Purple/Yellow
]
HOMOGRAPHY_FILE = 'homographies.json'
MERGE_THRESHOLD = 30

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


def extract_numbers(input_string):
    """Extracts four floating-point numbers from the input string."""
    pattern = r"[-+]?\d*\.\d+"
    matches = re.findall(pattern, input_string)
    return [float(match) for match in matches[:4]]

def parse_tracking_data(data):
    frames = defaultdict(list)
    for line in data.strip().split('\n'):
        parts = line.split(': ')
        frame_num = int(parts[0].split()[1])
        color = parts[1].split(',')[0]
        bbpoints = extract_numbers(parts[1])
        center_x = (bbpoints[0] + bbpoints[2]) / 2
        center_y = ((bbpoints[1] + bbpoints[3]*3) / 4)
        frames[frame_num].append({'color': color, 'point': (center_x, center_y)})
    return frames


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
cv2.namedWindow('Tracking Visualization')
cv2.setMouseCallback('Tracking Visualization', mouse_callback)
frame_nums = sorted(tracking_data.keys())
enable_merging = False

for frame_num in frame_nums:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, video_frame = cap.read()
    if not ret:
        break

    frame_data = tracking_data.get(frame_num, [])
    if enable_merging:
        frame_data = merge_objects(frame_data, calibration_views)

    # Draw video objects
    for obj in frame_data:
        color = COLOR_SCHEMES[0][obj['color']]  # Original colors for video
        center = tuple(map(int, obj['point']))
        cv2.circle(video_frame, center, 5, color, -1)

    # Draw map projections for active views
    map_display = map_image_resized.copy()
    for view_idx, view in enumerate(calibration_views):
        if not view['active']:
            continue
            
        color_scheme = COLOR_SCHEMES[view_idx % len(COLOR_SCHEMES)]
        for obj in frame_data:
            point = obj['point']
            if cv2.pointPolygonTest(view['src_points'], point, False) >= 0:
                src_pt = np.array([[point]], dtype=np.float32)
                dst_pt = cv2.perspectiveTransform(src_pt, view['homography'])
                map_point = tuple(map(int, dst_pt[0][0]))
                cv2.circle(map_display, map_point, 5, color_scheme[obj['color']], -1)

    # Draw UI elements
    status_text = "Merging: ON" if enable_merging else "Merging: OFF"
    cv2.putText(map_display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    
    # Draw view activation status
    for idx, view in enumerate(calibration_views[:3]):
        y_pos = 60 + idx*30
        color = COLOR_SCHEMES[idx % len(COLOR_SCHEMES)]['blue']
        status = f"View {idx+1}: {'ON' if view['active'] else 'OFF'}"
        cv2.putText(map_display, status, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
    key = cv2.waitKey(int(1000/FPS)) & 0xFF
    if key == ord('n'):
        # Start new calibration - pause video and show static frame
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

            combined_cal = np.hstack((temp_vid, temp_map))
            cv2.imshow('Tracking Visualization', combined_cal)
            
            k = cv2.waitKey(1)
            if k == 27 or current_state == IDLE:
                if len(video_points) == 4 and len(map_points) == 4:
                    src_pts = np.array(video_points, dtype=np.float32).reshape(-1,1,2)
                    dst_pts = np.array(map_points, dtype=np.float32).reshape(-1,1,2)
                    homography, _ = cv2.findHomography(src_pts, dst_pts)
                    calibration_views.append({
                        'src_points': np.array(video_points, dtype=np.int32),
                        'dst_points': np.array(map_points, dtype=np.int32),
                        'homography': homography,
                        'active':True
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
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()