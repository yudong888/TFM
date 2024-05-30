import cv2

def cut_video(input_video, output_video, start_frame, end_frame):
    # 1.源视频处理
    capture = cv2.VideoCapture(input_video)  # 首先获取并打开源视频，这个自己弄好路径就好
    
    # 2.创建一个写入视频对象
    output = output_video  # 这是我们要保存的一小段视频的文件路径，要精确到文件名
    # 下面三个cap.get我加了int()强制类型转换，因为返回的是float类型，在创建写入视频对象时不允许，也有可能有的opencv版本不需要，大家可以自行尝试判断，加一个不碍事
    fps = int(capture.get(cv2.CAP_PROP_FPS)) # 获取视频帧数，或者自己写
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) # 获取视频宽，或者自己写
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 获取视频长，或者自己写
    print("fps:", fps)
    print("width:", width)
    print("height:", height)
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # 用于avi格式的生成的参数
    # fourcc = cv2.VideoWriter_fourcc('X','V','I','D')  # 用于avi格式的生成的参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成的参数
    videowriter = cv2.VideoWriter(output, fourcc, fps, (width, height))  # 创建一个写入视频对象

    # 3.开始保存目标视频
    print('Start!!!')
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 设置开始帧
    pos = capture.get(cv2.CAP_PROP_POS_FRAMES)  # 获取当前帧数
    while pos < end:  # 从start到end之间读取帧数
        ret, frame = capture.read()  # 从开始帧开始读取，之后会从开始帧依次往后读取，直到退出循环
        videowriter.write(frame)  # 利用'写入视频对象'写入帧
        pos = capture.get(cv2.CAP_PROP_POS_FRAMES)  # 获取当前帧数pos
    videowriter.release()  # 关闭写入视频对象
    print('Finished!!!')
    capture.release()  # 关闭读取视频对象

if __name__ == '__main__':
    start = 2430
    end = 2500
    file_name = '2024-01-05T09_43_39Z-left'
    file_type = '.mp4'
    input_file = file_name + file_type
    output_file = file_name + '_' + str(start) + '_' + str(end) + file_type
    cut_video(input_video=input_file, output_video=output_file, start_frame=start, end_frame=end)