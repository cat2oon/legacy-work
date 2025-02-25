import cv2


def video_to_frames(video_path, save_dir):
    assert video_path is not None, "video_path required"

    video_capture = cv2.VideoCapture(video_path)
    success, image = video_capture.read()

    count = 0
    while success:
        save_path = "{}/frame{}.jpg".format(save_dir, count)
        cv2.imwrite(save_path, image)
        success, image = video_capture.read()
        if success is not True:
            print('>>> failed to read frame [{}]'.format(count))
        count += 1

    return count
