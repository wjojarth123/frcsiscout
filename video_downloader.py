from config import video_paths
from pytubefix import YouTube
from tqdm import tqdm
import os

def clean(folder_path):
    if not os.path.exists(folder_path):
        print("Folder does not exist.")
        return

    for file_name in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, file_name))

def main(): 
    clean("videos/raw/")
    for video_path in tqdm(video_paths):
        yt = YouTube(video_path)
        stream = yt.streams.filter(only_video=True).order_by('bitrate').desc().first()
        stream.download('videos/raw/')


if __name__ == "__main__": 
    main()