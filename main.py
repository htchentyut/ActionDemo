# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from person_box import person_box
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    video_image_path = '/Users/"USERNAME"/data/P19/image/'
    output_path = '/Users/"USERNAME"/demo/'
    videonames = os.listdir(video_image_path)
    for videoname in videonames:
        image_path = os.path.join(video_image_path, videoname)
        keypoints_path = os.path.join(output_path, videoname)
        person_box(image_path, keypoints_path)
    # person_box()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



