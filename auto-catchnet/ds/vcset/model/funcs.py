import os
import cv2
import glob
import re


def grep_all_videos(source_base_path):
    sub_list = [os.path.join(source_base_path, dir_name) for dir_name in os.listdir(source_base_path)]
    dir_list = [sub_path for sub_path in sub_list if os.path.isdir(sub_path)]

    vid_paths = glob.glob("{}/*.mp4".format(source_base_path))
    vids = [get_uid_from_video_path(vid_path) for vid_path in vid_paths]
    vids = [v for v in vids if v is not None]

    if len(dir_list) is 0:
        return vids

    sub_vids = []
    for dir_path in dir_list:
        subs = grep_all_videos(dir_path)
        sub_vids += subs

    return vids + sub_vids


def get_uid_from_video_path(vid_path):
    uid_pattern = "(?P<uid>\w+)\/(?:\d\/)?(?:[\w-]*?record-.+\.mp4)"
    m = re.search(uid_pattern, vid_path)

    if m is None:
        print(vid_path)
        return None

    vid_uid = m.group("uid")
    return vid_uid, vid_path


def video_to_frames(uid_vid, save_base_path):
    uid, video_path = uid_vid
    assert video_path is not None, "video_path required"

    filename = os.path.basename(video_path)
    filename = os.path.splitext(filename)[0]
    save_dir_path = "{}/{}/{}".format(save_base_path, uid, filename)

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    video_capture = cv2.VideoCapture(video_path)
    success, image = video_capture.read()

    count = 0
    while success:
        save_path = "{}/frame-{:05d}.jpg".format(save_dir_path, count)
        cv2.imwrite(save_path, image)
        success, image = video_capture.read()
        count += 1
        if success is not True:
            print('>>> {} exit frame [{}]'.format(save_dir_path, count))

    return count


if __name__ == '__main__':
    base_path = "/home/chy/archive-data/bench/mturk"
    save_base_path = "/home/chy/archive-data/processed/vc-one/frames"

    for uv in grep_all_videos(base_path):
        video_to_frames(uv, save_base_path)

