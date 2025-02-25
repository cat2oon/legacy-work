from tensorflow.compat.v2.data import Dataset
from tensorflow.python.ops import random_ops

import tensorflow as tf
import time, os, json

frame_shuffle_buffer = 900
interleave_cycle = 10

class DatasetReader():
    image_feature_description = {
        'subject': tf.io.FixedLenFeature([], tf.string),
        'frame': tf.io.FixedLenFeature([], tf.string),
        'img_re': tf.io.VarLenFeature(tf.string),
        'img_le': tf.io.VarLenFeature(tf.string),
        'img_we': tf.io.VarLenFeature(tf.string),
        'gaze': tf.io.FixedLenFeature([6], tf.float32),
        'eyes': tf.io.FixedLenFeature([6], tf.float32),
        'poses': tf.io.FixedLenFeature([9], tf.float32),
        'rR': tf.io.FixedLenFeature([9], tf.float32),
        'lR': tf.io.FixedLenFeature([9], tf.float32),
        'cR': tf.io.FixedLenFeature([9], tf.float32),
        'gaze2d': tf.io.FixedLenFeature([3], tf.float32),
    }

    def __init__(self, config, phase, epoch, subject_batched=True, batch_number=-1):    
        self.phase = phase
        self.config = config
        tfrecord_path = os.path.join(config.data_path, phase)

        if subject_batched:
            temp_dataset = self._make_subject_batched_record(tfrecord_path, config)
        else :
            temp_dataset = self._make_random_batched_record(tfrecord_path, config)
#         self.length = -1
        if batch_number > 0:
            self.length = batch_number
        else:
            self.length = self._calculate_length(config, phase, subject_batched)
        
        self.dataset = temp_dataset.repeat(epoch)

    def __len__(self):
        return self.length
    
    def _calculate_length(self, config, phase, subject_batched):
        with open(os.path.join(config.data_path, phase, 'files.json'), 'r') as f:
            data_json = json.load(f)['subjects']

        if subject_batched:
            number_batch = 0
            
            for subject in data_json.keys():
                gaze_targets = data_json[subject]
                
                num_frames = 0
                for target in gaze_targets:
                    num_frames += len(data_json[subject][target])

                number_batch += num_frames // config.batch_size
            return number_batch
        else :
            num_total = 0
            
            for subject in data_json.keys():
                gaze_targets = data_json[subject]
                
                num_frames = 0
                for target in gaze_targets:
                    num_frames += len(data_json[subject][target])

                num_total += num_frames
            return num_total // config.batch_size

    def _make_subject_batched_record(self, tfrecord_path, config):
        # Make TFRecord File list and shuffle
        tfrecord_data = tf.data.Dataset.list_files(tfrecord_path+'/*.tfrecords')

        # Shuffle Batch (not exactly but efficient)
        tfrecord_data = tfrecord_data.interleave(
            lambda x : self._parse_one_record(x, config, tf.data.experimental.AUTOTUNE).batch(config.batch_size, 
                                                                                              drop_remainder=True),
            cycle_length=interleave_cycle, 
            block_length=1).prefetch(tf.data.experimental.AUTOTUNE)
        return tfrecord_data

    def _make_random_batched_record(self, tfrecord_path, config):
        # Make TFRecord File list and shuffle
        tfrecord_data = tf.data.Dataset.list_files(tfrecord_path+'/*.tfrecords')

        # Shuffle Batch (not exactly but efficient)
        tfrecord_data = tfrecord_data.interleave(
            lambda x : self._parse_one_record(x, config, 1),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            cycle_length=interleave_cycle, 
            block_length=1).prefetch(tf.data.experimental.AUTOTUNE).batch(config.batch_size,drop_remainder=True)
        return tfrecord_data
        
    
    # Parse one TFRecord File and split into batches
    def _parse_one_record(self, record_name, config, parallel):
        raw_image_dataset = tf.data.TFRecordDataset(record_name)
        parsed_image_dataset = raw_image_dataset.map(lambda x : self._preprocess(x, config), 
                                                     num_parallel_calls=parallel)
        parsed_image_dataset = parsed_image_dataset.shuffle(frame_shuffle_buffer)
        return parsed_image_dataset

    # Parse TFRecordfile
    def _parse_features(self, example_raw):
      # Parse the input tf.Example proto using the dictionary above.
      return tf.io.parse_single_example(example_raw, DatasetReader.image_feature_description)

    # PreProcess tf.Example to tf.data.Dataset 
    def _preprocess(self, record, config):
        record = self._parse_features(record)
        
        img_re = record['img_re'].values[0]
        img_re = tf.image.decode_jpeg(img_re)[16:-16,:,::-1]
        img_re = tf.image.convert_image_dtype(img_re, tf.float32)
#         img_re = tf.image.resize(img_re, [config.img_size_y, config.img_size_x])

        img_le = record['img_le'].values[0]
        img_le = tf.image.decode_jpeg(img_le)[16:-16,:,::-1]
        img_le = tf.image.convert_image_dtype(img_le, tf.float32)
#         img_le = tf.image.resize(img_le, [config.img_size_y, config.img_size_x])

        img_we = record['img_we'].values[0]
        img_we = tf.image.decode_jpeg(img_we)[:,:,::-1]
        img_we = tf.image.convert_image_dtype(img_we, tf.float32)
#         img_we = tf.image.resize(img_we, [config.whole_eye_image_y, config.whole_eye_image_x])

        right_gaze = record['gaze'][0:3]
        left_gaze = record['gaze'][3:6]
        gaze_cm = record['gaze2d'] / 10

        if self.phase == 'train':
            img_re, img_le, img_we = self._add_random_noise(img_re, img_le, img_we)

        return (img_re, img_le, img_we, record['eyes'], record['poses'], # 학습에 필수적인 부분들
                record['rR'], record['lR'], record['cR'], record['gaze2d'], record['gaze2d']),\
               (gaze_cm, gaze_cm, gaze_cm)

    def _add_random_noise_each(self, img):
        img = self._add_gaussian_noise(img, 0.005)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.8, 1.0)
        img = tf.image.random_hue(img, 0.05)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        
        return tf.clip_by_value(img, 0.0, 1.0)    
    
    def _add_random_noise(self, img1, img2, img3):
        img1 = self._add_gaussian_noise(img1, 0.005)
        img2 = self._add_gaussian_noise(img2, 0.005)
        img3 = self._add_gaussian_noise(img3, 0.005)

        img1, img2, img3 = self._add_random_brightness(img1, img2, img3, 0.15)
        img1, img2, img3 = self._add_random_contrast(img1, img2, img3, 0.8, 1.0)
        img1, img2, img3 = self._add_random_hue(img1, img2, img3, 0.05)
        img1, img2, img3 = self._add_random_saturation(img1, img2, img3, 0.8, 1.2)
        return tf.clip_by_value(img1, 0.0, 1.0), tf.clip_by_value(img2, 0.0, 1.0), tf.clip_by_value(img3, 0.0, 1.0)
    
    def _add_gaussian_noise(self, img, std):
        return img + tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=std, dtype=tf.float32)
    
    def _add_random_brightness(self, img1, img2, img3, max_delta):
        delta = random_ops.random_uniform([], -max_delta, max_delta)
        return tf.image.adjust_brightness(img1, delta), tf.image.adjust_brightness(img2, delta), tf.image.adjust_brightness(img3, delta)
    
    def _add_random_contrast(self, img1, img2, img3, lower, upper):
        contrast_factor = random_ops.random_uniform([], lower, upper)
        return tf.image.adjust_contrast(img1, contrast_factor), tf.image.adjust_contrast(img2, contrast_factor), tf.image.adjust_contrast(img3, contrast_factor)
    
    def _add_random_hue(self, img1, img2, img3, max_delta):
        delta = random_ops.random_uniform([], -max_delta, max_delta)
        return tf.image.adjust_hue(img1, delta), tf.image.adjust_hue(img2, delta), tf.image.adjust_hue(img3, delta)

    def _add_random_saturation(self, img1, img2, img3, lower, upper):
        saturation_factor = random_ops.random_uniform([], lower, upper)
        return tf.image.adjust_saturation(img1, saturation_factor), tf.image.adjust_saturation(img2, saturation_factor), tf.image.adjust_saturation(img3, saturation_factor)