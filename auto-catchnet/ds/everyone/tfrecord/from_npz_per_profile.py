import argparse
import glob
import os

import numpy as np
import tensorflow as tf

from ac.common.images import byte_arr_to_img
from ds.everyone.model.item_index import ItemIndex
from ds.everyone.profile.profiles import get_profile_ids
from ds.core.tf.features import feature_int64, feature_bytes, feature_float
from ds.core.tf.tfrecords import make_writer
from ai.feature.face.candide import Candide

candide = Candide("/home/chy/archive-models/candide/candide.npz",
                  "/home/chy/archive-models/candide/shape_predictor_68_face_landmarks.dat")


def get_candide_feature(encoded_img):
    img = byte_arr_to_img(encoded_img)
    return candide.get_features(img)


def load_npz_from(data_path, block_id):
    npz_path = os.path.join(data_path, "item-{:05d}.npz".format(block_id))
    return load_npz(npz_path)


def load_npz(npz_path):
    npz = np.load(npz_path)
    images = npz['images']
    indexes = npz['metas']
    return ItemIndex.from_npz(indexes, images)


def to_tfexample(item_index: ItemIndex, extract_feature=False):
    uid = item_index.uid.astype(np.int64)
    so = item_index.orientation.astype(np.int64)
    cx = item_index.camera_x.astype(np.float)
    cy = item_index.camera_y.astype(np.float)
    csx = item_index.camera_to_screen_x.astype(np.float)
    csy = item_index.camera_to_screen_y.astype(np.float)

    ife = item_index.image_frame
    iff = item_index.image_face
    ile = item_index.image_left_eye
    ire = item_index.image_right_eye

    candide = ""
    if extract_feature:
        candide = get_candide_feature(ife)

        if candide is None:
            return None

    # TODO: image binary를 제외한 meta 정보는
    # uid만 저장하고 그 외는 meta_repository를 통해 접근
    # NOTE: 이미지는 모두 jpeg encoded 상태
    features = tf.train.Features(feature={
        'image/frame': feature_bytes(ife.tostring()),
        'image/face': feature_bytes(iff.tostring()),
        'image/left_eye': feature_bytes(ile.tostring()),
        'image/right_eye': feature_bytes(ire.tostring()),
        'model/candide': feature_bytes(candide.tostring()),
        'label/uid': feature_int64(uid),
        'label/orientation': feature_int64(so),
        'label/cam_x': feature_float(cx),
        'label/cam_y': feature_float(cy),
        'label/cam_to_screen_x': feature_float(csx),
        'label/cam_to_screen_y': feature_float(csy),
    })

    example = tf.train.Example(features=features)
    return example


def write_tfexample(item, writer):
    if item.is_pad():
        return
    try:
        example = to_tfexample(item, extract_feature=True)
        if example is not None:
            writer.write(example.SerializeToString())
    except Exception as e:
        print('UID [{}] exception: {}'.format(item.uid, e))


def convert(npz_path, out_path):
    print("*** convert ***")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    profile_ids = get_profile_ids()
    blocks = glob.glob(os.path.join(npz_path, "item-*.npz"))

    # makes writer per profile
    writer_per_profile = {}
    for profile_id in profile_ids:      # tfrecord-{profile_id}.tfr
        writer_per_profile[profile_id] = make_writer(out_path, int(profile_id))

    # per block loop
    for i, block_path in enumerate(blocks):
        item_indexes = load_npz(block_path)

        for item_index in item_indexes:
            writer = writer_per_profile[item_index.pid]     # pid 타입이 어찌되는지 체크
            write_tfexample(item_index, writer)

    # close all writer per profile
    for profile_id in profile_ids:
        writer = writer_per_profile[profile_id]
        writer.close()

    print("*** complete ***")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()

    convert(args.npz_path, args.out_path)
