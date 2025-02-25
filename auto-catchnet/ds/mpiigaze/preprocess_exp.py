import os
import argparse
import pandas as pd
import scipy.io

from common.images import make_dual_channel_image
from common.numerics import *


def get_eval_info(subject_id, eval_dir):
    df = pd.read_csv(os.path.join(eval_dir, '{}.txt'.format(subject_id)),
                     delimiter=' ', header=None, names=['path', 'side'])
    df['day'] = df.path.apply(lambda path: path.split('/')[0])
    df['filename'] = df.path.apply(lambda path: path.split('/')[1])
    df = df.drop(['path'], axis=1)
    return df


def get_subject_data(subject_id, data_dir, eval_dir):
    left_images, right_images = {}, {}
    file_names, co_poses, co_gazes = {}, {}, {}

    dir_path = os.path.join(data_dir, subject_id)
    for name in sorted(os.listdir(dir_path)):
        path = os.path.join(dir_path, name)
        mat_data = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)

        data = mat_data['data']
        day = os.path.splitext(name)[0]
        co_poses[day] = data.left.pose
        co_gazes[day] = data.left.gaze
        left_images[day] = data.left.image
        right_images[day] = data.right.image
        file_names[day] = mat_data['filenames']

        if not isinstance(file_names[day], np.ndarray):
            co_poses[day] = np.array([co_poses[day]])
            co_gazes[day] = np.array([co_gazes[day]])
            left_images[day] = np.array([left_images[day]])
            right_images[day] = np.array([right_images[day]])
            file_names[day] = np.array([file_names[day]])

    images, poses, gazes = [], [], []
    for _, row in get_eval_info(subject_id, eval_dir).iterrows():
        day = row.day
        index = np.where(file_names[day] == row.filename)[0][0]

        image_l = left_images[day][index]
        image_r = right_images[day][index][:, ::-1]
        image = make_dual_channel_image(image_l, image_r)
        pose = convert_pose(co_poses[day][index])
        gaze = convert_gaze(co_gazes[day][index])

        images.append(image)
        poses.append(pose)
        gazes.append(gaze)

    images = np.array(images).astype(np.float32) / 255
    poses = np.array(poses).astype(np.float32)
    gazes = np.array(gazes).astype(np.float32)

    return images, poses, gazes


def load_subject(dataset_dir, subject_id):
    subject_id = 'p{:02}'.format(subject_id)
    data_dir = os.path.join(dataset_dir, 'Data', 'Normalized')
    eval_dir = os.path.join(dataset_dir, 'Evaluation Subset', 'sample list for eye image')
    return get_subject_data(subject_id, data_dir, eval_dir)


def save_to_npz(out_dir, subject_id, images, poses, gazes):
    out_path = os.path.join(out_dir, "p{:03}-exp".format(subject_id))
    np.savez(out_path, image=images, pose=poses, gaze=gazes)


def to_npz(dataset_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for subject_id in range(15):
        images, poses, gazes = load_subject(dataset_dir, subject_id)
        save_to_npz(out_dir, subject_id, images, poses, gazes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    to_npz(args.dataset, args.outdir)

