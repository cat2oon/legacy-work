import multiprocessing as mp
import os
from multiprocessing import JoinableQueue
from multiprocessing.context import Process
from multiprocessing.pool import Pool

import cv2
import numpy as np

from ac.common.jsons import load_json
from ds.vcset.model.funcs import grep_all_videos

"""
 - Fast parallel processing framer
"""


def archive_item(queue: JoinableQueue):
    while True:
        item = queue.get()

        if item == 5:
            print("*** archiver task done ***")
        else:
            print(item["filename"])
        queue.task_done()


def video_to_item(uid_and_path, queue: JoinableQueue):
    uid, video_path = uid_and_path
    print(">>>>>> {}".format(uid))

    if uid == 5:
        queue.put(5)
    else:
        queue.put({'filename': video_path})
    return

    uid, video_path = uid_and_path
    filename = os.path.basename(video_path)
    filename = os.path.splitext(filename)[0]
    meta_path = video_path.replace('mp4', 'meta')

    item_meta = load_json(meta_path)
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()

    frames = [frame]
    while success:
        success, frame = video_capture.read()

    queue.put({
        "uid": uid,
        "filename": filename,
        "meta": item_meta,
        "frames": np.asarray(frames),
    })


class Framer:
    def __init__(self,
                 source_base_path,
                 archive_base_path,
                 num_archiver=4,
                 num_producer=16):
        self.video_index = None
        self.num_producer = num_producer
        self.num_archiver = num_archiver
        self.source_base_path = source_base_path
        self.archive_base_path = archive_base_path
        self.item_to_archive = None

        self.build_managed_queue()
        self.scan_videos()
        self.set_terminator_tail()

    def build_managed_queue(self):
        manager = mp.Manager()
        self.item_to_archive = manager.JoinableQueue()

    def scan_videos(self):
        self.video_index = grep_all_videos(self.source_base_path)

    def set_terminator_tail(self):
        # terminator_tail = NULL_ITEM * self.num_archiver
        terminator_tail = [(5, ""),(5, ""),(5, ""),(5, "")]
        self.video_index = self.video_index + terminator_tail

    def to_npz_all(self):
        queue = self.item_to_archive
        for i in range(self.num_archiver):
            process = Process(target=archive_item, args=(queue,))
            process.start()

        with Pool(self.num_producer) as p:
            for v in self.video_index:
                p.apply(video_to_item, (v, queue))

        self.item_to_archive.join()
        print("*** complete ***")
