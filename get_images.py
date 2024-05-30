import cv2
import os
import sys
import re


def get_files(folder_path, file_type, keyword):
    # files = os.listdir(folder_path)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(folder_path+f)]
    video_list = []
    print('Files for extracting images: ')
    for filename in files:
        sep=re.split('[-_.]', filename)
        if sep[-1]==file_type and sep[-2]==keyword:
            video_list.append(filename)
    return video_list

def get_images(video_path, video_name, image_path, interval):
    # configure the paths
    video_path = video_path + video_name

    # generate a folder
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    
    # read video
    if not os.path.isfile(video_path):
        print(video_path + ' no exist')
        sys.exit(1)
    cap = cv2.VideoCapture(video_path)
    has_frame = True
    idx = 0
    n = 0
    print('Start extracting images', video_name)
    while has_frame:
        has_frame, frame = cap.read()
        if has_frame and n==interval:
            file_name = f'{video_name.split(".")[0]}{idx:06d}.png'
            cv2.imwrite(os.path.join(image_path, file_name), frame)
            n = 0
        idx += 1
        n += 1
    cap.release()
    print('Done processing')

if __name__ == '__main__':
    # configure the paths
    # folder_path = 'C:/Users/yy19860/Downloads/PBvideos/OldCamera/'
    folder_path = './'
    file_type = 'mp4'
    key_word = 'left'
    output_path = folder_path +'images/'
    multiplevideos = False
    interval = 10
    # output_path = folder_path +'images/' + input_video.split('.')[0] + '/'
    if multiplevideos:
        video_list = get_files(folder_path, file_type, key_word)
        # print(video_list)
        for input_video in video_list:
            get_images(folder_path, input_video, output_path, interval)
    else:
        input_video = '2023-09-27T10_43_08Z-bottom.mp4'
        output_path = output_path + input_video.split(".")[0] + '/'
        get_images(folder_path, input_video, output_path, interval)
