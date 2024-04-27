# main.py
from utilities.counter import PersonCounter
from utilities.people_detector import detect_persons_in_video
from utilities.people_detector import detect_persons_in_image


if __name__ == "__main__":
    video_path1 = "Videos/video1.mp4"
    video_path2 = "Videos/video2.mp4"
    image_path1 = "Images/Group.jpg"

    # Initialize the person counter
    counter = PersonCounter()

    detect_persons_in_video(video_path1, counter)
    detect_persons_in_video(video_path2, counter)
    detect_persons_in_image(image_path1, counter)0
