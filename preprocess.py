from tqdm import tqdm
import os
import glob
import cv2
import apriltag
import numpy as np
import config 

def clean(folder_path):
    if not os.path.exists(folder_path):
        print("Folder does not exist.")
        return

    for file_name in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, file_name))
        
def get_center(results, tag_id):
    for result in results:
        if (result.tag_id == tag_id):
            return np.mean(result.corners, axis=0)
    return None
    

def center_video(video_path):
    options = apriltag.DetectorOptions(families='tag36h11')
    detector = apriltag.Detector(options)
    
    bottom_right_center = []
    bottom_left_center = []

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    while ret: 
        bottom_left = frame[400:, :640, :]
        bottom_right = frame[400:, 640:, :]

        bottom_left_gray = cv2.cvtColor(bottom_left, cv2.COLOR_BGR2GRAY)
        bottom_right_gray = cv2.cvtColor(bottom_right, cv2.COLOR_BGR2GRAY)

        bottom_right_results = detector.detect(bottom_right_gray)
        bottom_left_results = detector.detect(bottom_left_gray)

        bottom_left_center.append(get_center(bottom_left_results, 21))
        bottom_right_center.append(get_center(bottom_right_results, 10))

        ret, frame = cap.read()

    bottom_right_center = list(filter(lambda x: x is not None, bottom_right_center))
    bottom_left_center = list(filter(lambda x: x is not None, bottom_left_center))

    bottom_right_center = np.array(bottom_right_center)
    bottom_left_center = np.array(bottom_right_center)

    bottom_right_center = np.mean(bottom_right_center, axis=0)
    bottom_left_center = np.mean(bottom_left_center, axis=0)

    return None, bottom_right_center, bottom_left_center

def align_image(image, target_center, tag_center):
    if (tag_center is None):
        return None

    dx = int(target_center[0] - tag_center[0])
    dy = int(target_center[1] - tag_center[1])

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    centered_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return centered_image

def process_video(video_path):
    # top_center, bottom_right_target, bottom_left_target, = center_video(video_path)

    options = apriltag.DetectorOptions(families='tag36h11')
    detector = apriltag.Detector(options)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    top_images = []
    bottom_right_images = []
    bottom_left_images = []
    
    while ret: 
        top = frame[:400, :, :]
        bottom_right = frame[400:, 640:, :]
        bottom_left = frame[400:, :640, :]

        bottom_right_gray = cv2.cvtColor(bottom_right, cv2.COLOR_BGR2GRAY)
        bottom_left_gray = cv2.cvtColor(bottom_left, cv2.COLOR_BGR2GRAY)

        bottom_right_results = detector.detect(bottom_right_gray)
        bottom_left_results = detector.detect(bottom_left_gray)

        bottom_right_center = get_center(bottom_right_results, 10)
        bottom_left_center = get_center(bottom_left_results, 21)

        top_images.append(top)
        bottom_right_images.append(align_image(bottom_right, config.bottom_right_target, bottom_right_center))
        bottom_left_images.append(align_image(bottom_left, config.bottom_left_target, bottom_left_center))

        ret, frame = cap.read()
    
    top_images = list(filter(lambda x: x is not None, top_images))
    bottom_right_images = list(filter(lambda x: x is not None, bottom_right_images))
    bottom_left_images = list(filter(lambda x: x is not None, bottom_left_images))

    return top_images, bottom_right_images, bottom_left_images

def  write_images(images, output_path, type):
    os.makedirs("videos/clean/" + type, exist_ok=True)
    for idx, image in enumerate(images):
        if (idx > 400):
            print(f"videos/clean/{type}/{output_path}_{idx}.png")
            cv2.imwrite(f"videos/clean/{type}/{output_path}_{idx}.png", image)

def main():
    clean("videos/clean/top")
    clean("videos/clean/bottom_right")
    clean("videos/clean/bottom_left")
    for video_path in tqdm(glob.glob("videos/raw/*")):
        print(video_path.replace("videos/raw/", "").replace(".mp4", ""))
        top_images, bottom_right_images, bottom_left_images = process_video(video_path)
        write_images(top_images, video_path.replace("videos/raw/", "").replace(".mp4", ""), "top")
        write_images(bottom_right_images, video_path.replace("videos/raw/", "").replace(".mp4", ""), "bottom_right")
        write_images(bottom_left_images, video_path.replace("videos/raw/", "").replace(".mp4", ""), "bottom_left")
        break

if __name__ == "__main__": 
    main()
    
    